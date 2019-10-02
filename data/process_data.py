import sys
import collections
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # LOAD
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # TRANSFORM
    # catch duplicate ids, duplicate messages should be removed as might sugges inaccuracies while acquiring data
    # messages
    msg_list = messages.id.values
    mes_dup_id = [item for item, count in collections.Counter(msg_list).items() if count > 1]
    mes_dup    = messages[messages['id'].isin(mes_dup_id)]
    # cats
    cat_list = categories.id.values
    cat_dup_id = [item for item, count in collections.Counter(cat_list).items() if count > 1]
    cat_dup    = categories[categories['id'].isin(cat_dup_id)]
    print('Difference in duplicates between the two datasets: ',set(mes_dup_id) - set(cat_dup_id))
    # drop dulpicate ids which will create problems afterwards
    messages.drop_duplicates(subset='id',inplace=True)
    categories.drop_duplicates(subset='id',inplace=True)
    # store ids and merge 
    cat_ids = categories['id']
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    # select the first row of the categories dataframe
    categories = df['categories'].copy().str.split(';',expand=True)
    cat_ids = df['id']
    row = categories.iloc[0,:].apply(lambda x:x.split('-')[0])
    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.unique())
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    categories['id'] = cat_ids
    
    # drop the original categories column from `df`, add new categories and drop duplicates
    df.drop(columns=['categories'], inplace=True)    
    df = df.merge(categories, on='id')
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    #'mysql+mysqldb://'+
    engine = create_engine('sqlite:///'+database_filename)
#     engine = create_engine(database_filename)
    sql_table_name = 'disaster_msg'
    try:
        df.to_sql(sql_table_name, engine, index=False)
        print('Success: table {} created'.format(sql_table_name))
    except Exception as e:
        print('Table not created:',e)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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