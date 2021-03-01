#This file contains a program that will intake movie data, transform it, and load it into a SQL database

#imports
import json
import pandas as pd
import numpy as np
import re

import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
import time

#postgres password
from config import db_password

#file directory
file_dir = 'C:/Users/raywh/Class/whelan_etl/Data'

#Postgres database address
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"

#Nested method that will load and clean the wiki movie data
#Method takes the file name as a parameter
# ASSUMPTION: Wikipedia movies JSON file is in the target directory
def load_wiki(wiki_data):
    #load the wikipedia data
    try:
        with open(f'{file_dir}/{wiki_data}', mode='r') as file:
            wiki_movies_raw = json.load(file)
    
    except IOError:
        print('File not located in target directory')

    #Remove non-movie entries
    wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

    #Cleans the Wikipedia movies dataset
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune-Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles

        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie

    #Call the clean movies method and convert the json data to a dataframe
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)

    #Drop duplicate entries
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    wiki_movies_df.head()

    #Remove unnecessary columns
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    #Clean the box office data
    box_office = wiki_movies_df['Box office'].dropna() 
    #box_office[box_office.map(lambda x: type(x) != str)]

    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illi?on)'
    #box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,','', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan

    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    #Clean the budget data
    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    #budget[~matches_form_one & ~matches_form_two]

    budget = budget.str.replace(r'\[\d+\]\s*', '')
    budget[~matches_form_one & ~matches_form_two]
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    #Clean the release dates
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    #Clean the run time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()
    running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]
    running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()
    running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    return wiki_movies_df

#Load and clean the kaggle metadata
#Method takes the file name as a parameter
# ASSUMPTION: kaggle metadata csv file is in the target directory
def load_kaggle(kaggle_data):
    #Import the kaggle data
    try:
        kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv')

    except IOError:
        print('File not located in target directory')

    #Clean the kaggle metadata
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    return kaggle_metadata

#Load and clean ratings data
#Method takes the file name as a parameter
# ASSUMPTION: ratings csv file is in the target folder
def load_ratings(ratings_data):
    
    try:
        ratings = pd.read_csv(f'{file_dir}/{ratings_data}')
    except IOError:
        print('File not located in target directory')

    #Clean and transform ratings data
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) \
                    .pivot(index='movieId',columns='rating', values='count')
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    return rating_counts


def merge_tables(wiki, kaggle, ratings):
    #Merge the tables
    movies_df = pd.merge(wiki, kaggle, on='imdb_id', suffixes=['_wiki','_kaggle'])

    #Fill in missing data and clean
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)
        
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]

    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)

    #Merge Ratings Data

    movies_with_ratings_df = pd.merge(movies_df, ratings, left_on='kaggle_id', right_index=True, how='left')
    movies_with_ratings_df[ratings.columns] = movies_with_ratings_df[ratings.columns].fillna(0)

    return movies_with_ratings_df

#Method to upload our transformed data to Postgres database
#Method receives the modified movies dataframe and the ratings file name as parameters
# ASSUMPTION: Both the movies data frame and the ratings csv file have the same number of columns as our target Postgres tables, 42 and 5 respectively
def to_Postgres(movies, ratings):
    # Connecting to Postgres
    engine = create_engine(db_string)

    # Upload the Movies data
    try:
        movies.to_sql(name='movies', con=engine, if_exists='append')
    except sqlalchemy.exc.ProgrammingError:
        print('Movies dataframe does not match target table')
    # Upload the Ratings data
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    
    try:
        for data in pd.read_csv(f'{file_dir}/{ratings}', chunksize=1000000):
            print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
            data.to_sql(name='ratings', con=engine, if_exists='append')
            rows_imported += len(data)

            # add elapsed time to final print out
            print(f'Done. {time.time() - start_time} total seconds elapsed')
    except IOError:
        print('File not located in target directory')
    except sqlalchemy.exc.ProgrammingError:
        print('Ratings dataframe does not match target table')

    return

#Call the methods to load our datasets
wiki_df = load_wiki('wikipedia.movies.json')
kaggle_df = load_kaggle('movies_metadata.csv')
ratings_df = load_ratings('ratings.csv')

#Call the method to merge our data
movies_df = merge_tables(wiki_df, kaggle_df, ratings_df)

#Call the method to load to Postgres
to_Postgres(movies_df, 'ratings.csv')