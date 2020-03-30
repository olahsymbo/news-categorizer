# Dependencies 
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
cate_dir = os.path.realpath(os.path.dirname(app_path))
import sys
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import tensorflow as tf
from urllib.request import Request, urlopen 
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
import category_output
from nltk.tokenize import word_tokenize


app = Flask(__name__)

@app.route('/categorize', methods=['POST'])
def categorize():
    
    if textmodel:
        datan = []
        try:
            json_ = request.json
            print(json_)
            query = json_
            
            for i in range(len(query)):
                url = (query[i])
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                soup = BeautifulSoup(webpage, "html5lib")
                for a in soup.find_all('a'): 
                    a.decompose()
                contt = soup.findAll('p')
                testdata = [re.sub(r'<.+?>',r'',str(x)) for x in contt]
                snh = ' '
                testdata = snh.join(testdata)
                linessn = testdata.strip()
                porter = PorterStemmer()
                liness = porter.stem(linessn)                
                Yb = tfidf_model.transform([liness])
                with graph1.as_default():
                     prediction = textmodel.predict(Yb)            
                     print(np.argmax(prediction))
                     datan.append(category_output.category_output(prediction))
                
            return jsonify(datan)

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 6000

    tfidf_model = joblib.load(open(os.path.join(cate_dir, "tfidfmodel.pkl")))
    print ('text vectorizer loaded')
    
    textmodel = load_model(open(os.path.join(cate_dir, "News_Cate.h5")))
    graph1 = tf.get_default_graph()
    print ('Model loaded')

    app.run(port=port, debug=True)
