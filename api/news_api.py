# Dependencies 
import os, sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
cate_dir = os.path.realpath(os.path.dirname(app_path))
sys.path.insert(0, cate_dir)

from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
import re
from bs4 import BeautifulSoup
import tensorflow as tf
from urllib.request import Request, urlopen 
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
from helpers import category_output

app = Flask(__name__)
tfidf_model = joblib.load(open(os.path.join(cate_dir, "../trained_models/tfidfmodel.pkl"), 'rb'))
print('text vectorizer loaded')

textmodel = load_model(open(os.path.join(cate_dir, "../trained_models/News_Cate.h5"), 'rb'))
graph1 = tf.get_default_graph()
print('Model loaded')

@app.route('/categorize', methods=['GET'])
def categorize():

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

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 6000

    app.run(port=port, debug=True)
