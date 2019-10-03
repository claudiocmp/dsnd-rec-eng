import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_msg', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # 2
    msg_len_df = pd.DataFrame(df['message'].apply(len))
    msg_len = msg_len_df.values
    l_min,l_max = min(msg_len), max(msg_len)

    def tukey_rule(df, col_name):
        mean = df[col_name].mean()
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = 1.5*(Q3-Q1)
        v_min = Q1 - IQR
        v_max = Q3 + IQR
        return df[(df[col_name]>v_min)&(df[col_name]<v_max)]
        
    cleaned = tukey_rule(msg_len_df,'message')['message'].values
    l_min,l_max = min(cleaned), max(cleaned)

    bins = 10
    step = (l_max-l_min)/bins
    thshds = np.arange(l_min,l_max+step,step)
    intervals = {i:[a,b] for i,(a,b) in enumerate(zip(thshds[:-1],thshds[1:]))}
    lengths = []
    for i,iv in intervals.items():
        l = len(np.where(np.logical_and(cleaned>iv[0],cleaned<=iv[1]))[0])
        if i==0:
            l+=1
        lengths.append(l)
    
    iv_names = [str(v)[1:-1] for v in intervals.values()]
    
    # 3
    categories = list(set(df.columns) - set(['id', 'message', 'original', 'genre']))
    qty=[]
    for col in categories:
        qty.append(df[col].sum())
    s = sorted(zip(categories,qty), key=lambda x:x[1], reverse=True)
    categories,qty = zip(*s)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=iv_names,
                    y=lengths
                )
            ],
            'layout':{
                'title': 'Distribution of message lengths',
                'yaxis':{
                    'title':'Count'
                },
                'xaxis':{
                    'title':'Message length (chars)'
                }
            }
        },
        {
            'data':[
                Bar(
                    y=categories,
                    x=qty,
                    orientation='h'
                )
            ],
            'layout':{
                'title': 'Messages per Category',
                'yaxis':{
                    'title':'Categories'
                },
                'xaxis':{
                    'title':'Count'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()