from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import string
import bz2file as bz2
import joblib as jb

with bz2.BZ2File('wv.joblib.bz2', 'rb') as f:
    td = jb.load(f)

with bz2.BZ2File('md.joblib.bz2', 'rb') as f:
    mp = jb.load(f)

def  clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    return text

def make_test_predictions(df):
    df.comment_text = df.comment_text.apply(clean_text)
    X_test = df.comment_text
    X_test_transformed = td.transform(X_test)
    y_test_pred = mp.predict_proba(X_test_transformed)
    result =  sum(y_test_pred[0])
    if result >=1 :
       return 1
    else :
       return 0
    
app = Flask(__name__)
CORS(app)

@app.route("/media/text", methods=['POST'])
def sanitize(): 
    val = request.get_json()
    val = val['comment']
    comment_text = val
    comment ={'comment_text':[comment_text]}
    comment = pd.DataFrame(comment)
    result = make_test_predictions(comment)
    if(result==0):
        return(jsonify({"msg": 1}))
    else:
        return(jsonify({"msg": 0}))

if __name__ == '__main__':
    app.run()
