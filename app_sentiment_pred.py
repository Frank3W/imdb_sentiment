import json

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


@app.route('/text', methods=['POST'])
def sentiment_pred():
    review_text = request.form['text']
    review_text = [review_text]
    
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
    print(r_dict)
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
