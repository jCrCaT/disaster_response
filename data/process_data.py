import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        DEF: Load data from csv file and merge the data from messages and categories
        INPUT:
            messages_filepath - Path where is the data related to messages on csv format
            categories_filepath Path where is the data related to categories on csv format
        OUTPUT:
            df - Dataframe that contains the messages and categories merged into one
    '''
    #Read the messages file
    messages = pd.read_csv(messages_filepath)
    #Read the category file
    categories = pd.read_csv(categories_filepath)
    #Merge messages and categories by id
    df = pd.merge(messages,categories,on=['id','id'],how='left')
    return df


def clean_data(df):
    '''
        DEF: Function used to clean de data. What is does is defined on the next points
            1.- Split categories by ";"
            2.- Asign the column names with the first row
            3.- drop unnecesary columns
            4.- Concat DF with the new columns (categories from tweets)
            5.- Get the sum of data by category that is duplicated
            6.- Drop dupplicates
        INPUT:
            df - Dataframe that is going to be procesed by the model
        OUTPUT:
            df - Dataframe procesed with the changes described on DEF
    '''
    #Split words by ; on categories
    categories = df['categories'].str.split(';',expand=True)
    #Read the first row to get the column names
    row = categories.head(1)
    #Remove the last two characters for each row to get the name of the columns
    category_colnames = row.apply(lambda x: x[0][slice(0,-2)])
    #Put category_colnames as list
    category_colnames = list(category_colnames)
    #Change the columns names of categories
    categories.columns = category_colnames
    #Convert the values of rows into 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    #Drop categories column
    df.drop('categories',axis=1,inplace=True)
    #Concat categories with df
    df = pd.concat([df,categories],axis=1)
    #Count duplicates by id
    df[df.groupby("id")['id'].transform('size') > 1]
    #drop duplicates
    df = df.drop_duplicates(subset=['id'])
    return df

def save_data(df, database_filename):
    '''
        DEF: Function used to save the model
        INPUT:
            df - Dataframe to be saved
            database_filename - Name of the database that is going to contain the df
        OUTPUT None
    '''
    #Create engine
    engine = create_engine('sqlite:///'+database_filename)
    #save df to SQL
    df.to_sql('dataframe', engine, index=False)
    


def main():
    '''
        DEF: Main function used to run the process_data
        INPUT:
            None
        OUTPUT:
            None
    '''
    if len(sys.argv) == 4:
        #Get files names
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        #Load data into DF
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        #Clean data
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        #Save the data to the .db file
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

