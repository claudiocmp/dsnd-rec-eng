import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_msg', engine)
    X = df['message']
    Y = df.drop(columns=['message','original','id','genre'])
    return X,Y,Y.columns


def tokenize(text):
    norm = re.sub('[^A-Za-z0-9]', ' ', text.lower())
    words = word_tokenize(norm)
    words = [w for w in words if w not in stopwords.words('english')]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return words


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    test_report = []
    y_test_pred = model.predict(X_test)
    for i,col in enumerate(y_test):
        # test
        test_report.append(
        classification_report(
            y_test[col].values,
            y_test_pred[:,i]
        )
        )
    [print(r,'\n--------------') for r in test_report]


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    