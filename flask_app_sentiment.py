"""Flask app for IMDB sentiment prediction.
"""

import json
import os
import sqlite3
import datetime

from flask import Flask, render_template
from flask import request
import torch

from src import textproc
from src import textencoder
from src import torchnet


############ Text Process and Model ##############
# paths for process and model
MODEL_PATH = './savedspace/torchfullnet'
PROC_JSON = os.path.join(MODEL_PATH, 'textproc.json')
MODEL_TOPOLOGY = os.path.join(MODEL_PATH, 'imdb_fullnet_topology.json')
MODEL_WEIGHTS = os.path.join(MODEL_PATH, 'imdb_fullnet_weights.pt')

# load process step
loaded_textproc = textproc.TextProc.from_load_wcount_pair(PROC_JSON)

# load trained model
model = torchnet.FullNet.from_modeltopology(MODEL_TOPOLOGY)
model.load_model_weights(MODEL_WEIGHTS)
model = model.eval()


############ database for inqueries ##############
DB_FNAME = 'app_sentiment_records.db'

# create sqlite database to record queries.
conn = sqlite3.connect(DB_FNAME)
sql_create_record_table = """ CREATE TABLE IF NOT EXISTS query_records (
                                    time text NOT NULL,
                                    query text,
                                    sentiment_score text
                                ); """

cursor = conn.cursor()
cursor.execute(sql_create_record_table)
conn.close()


############ flask app ##############
app = Flask(__name__)

@app.route('/text', methods=['POST'])
def sentiment_pred():
    raw_text = request.form['text']
    review_text = [raw_text]
    
    # process text
    review_text_processed, selected_word = loaded_textproc.process(text_corpus=review_text)
    word_encoder = textencoder.OneHotEncoder(selected_word)
    review_text_encoded = word_encoder.encode(review_text_processed)
    
    # get prediction
    with torch.no_grad():
        sentiment_score = model.prediction(review_text_encoded).cpu().numpy()
        
    # return positivity percentage as json response
    r_dict = {}
    pos_pct = "{:.2%}".format(sentiment_score[0][0])
    r_dict['pos_pct'] = pos_pct
    
    # insert record in database
    with sqlite3.connect(DB_FNAME) as conn:
        insert_record(conn, str(datetime.datetime.now()), raw_text, pos_pct)
    
    response = app.response_class(
        response=json.dumps(r_dict),
        status=200,
        mimetype='application/json'
    )
    
    return response

@app.route("/")
def index():
    return render_template("index.html")


############ helpers ##############
def insert_record(conn, time_str, query_str, sentiment_score_str):

    sql = ''' INSERT INTO query_records(time, query, sentiment_score)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (time_str, query_str, sentiment_score_str))
    conn.commit()
    return True


if __name__ == "__main__":
    app.run(port=9988, debug=True)
