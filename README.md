# Movie Data ETL Challenge

For this challenge we are tasked with creating an automated data Extraction, Transformation, and Loading pipeline with movie metadata and ratings data from Wikipedia and Kaggle. This task required us to clean and extract the necessary data points from messy data sources.

The multi-step approach I took for this project reshaped the code we initially created in Jupyter during the lesson into methods that can be called whenever our data is updated. I created 5 total methods: one each to load and clean our three data sets, one to merge and further transform our tables into one "movies" dataframe, and one to connect to our Postgres server and upload our 'movied' and 'ratings' tables when an update is needed.

## Assumptions
This automated data ETL pipeline relies on several assumptions:
 
 1. Our three data sets are located in the target directory.
 2. Our three data sets will continually match the original versions in shape and the names of necessary columns
 3. Each step of the process completes successfully so that the next can run.
 4. Our database has been established and is ready to be connected to.
 5. Our database password is up to date.
 6. Lastly, our final product data sets MUST match the target Postgres tables in the number of columns (42 and 5 respectively) and column names. 
