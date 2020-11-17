import json
import sqlite3
import datetime

import flask
from flask import Flask, render_template, url_for
from flask import request
import torch

import textproc
import torchnet

app = Flask(__name__)

# load process step
loaded_textproc = textproc.TextProc.from_load_wcount_pair('text_proc.json')

# load trained model
fullnet = torchnet.FullNet('relu', loaded_textproc.top_num+1, 12, 8, 1)
fullnet.load_model_weights('imdb_fullnet.pt')
fullnet = fullnet.eval()

# create sqlite database to record queries.
DB_FNAME = 'app_sentiment_records.db'
conn = sqlite3.connect(DB_FNAME)

sql_create_record_table = """ CREATE TABLE IF NOT EXISTS query_records (
                                    time text NOT NULL,
                                    query text,
                                    sentiment_score text
                                ); """

cursor = conn.cursor()
cursor.execute(sql_create_record_table)
conn.close()

def insert_record(conn, time_str, query_str, sentiment_score_str):

    sql = ''' INSERT INTO query_records(time, query, sentiment_score)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (time_str, query_str, sentiment_score_str))
    conn.commit()
    return True


@app.route('/text', methods=['POST'])
def sentiment_pred():
    raw_text = request.form['text']
    review_text = [raw_text]
    
    # process text
    review_text_processed, selected_word = loaded_textproc.process(text_corpus=review_text)
    word_encoder = textproc.WordEncoder(selected_word)
    review_text_encoded = word_encoder.onehot_encode(review_text_processed)
    review_text_tensor = torch.from_numpy(review_text_encoded).float().to(torchnet.get_model_device(fullnet))
    
    # get prediction
    with torch.no_grad():
        sentiment_score = torch.sigmoid(fullnet(review_text_tensor)).cpu().numpy()
        
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
    print(url_for('sentiment_pred'))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=9988, debug=True)
