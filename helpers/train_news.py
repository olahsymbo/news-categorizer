import os
import sys
import inspect
import pickle
import string

app_path = inspect.getfile(inspect.currentframe())
cate_dir = os.path.realpath(os.path.dirname(app_path))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.models import load_model

from data.dataloader import NewsDataLoader

main_data_source = 'data/bbc'

ndl = NewsDataLoader(main_data_source)
datan = ndl.load_all_news()
            
porter = PorterStemmer()
data = [porter.stem(word) for word in datan]

Target = np.concatenate((target_entertain, target_sport, target_politics, target_business, target_tech))

# splitting data into training and testing set
X_trainn, X_testt, y_train, y_test = train_test_split(data, Target, random_state=0, test_size=0.35)

enc_length = 3000
tfidf = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english', max_features = enc_length)
tfidf_model = tfidf.fit(X_trainn)

pickle.dump(tfidf_model, open(os.path.join(cate_dir, "../trained_models/tfidfmodel.pkl", "wb")))

# ------------------------------------------------------------------------------------------#
# create model
epoch = 50
batch = 150

# alternative model
text_model = Sequential()
text_model.add(Dense(500, input_dim=enc_length, activation='relu'))
text_model.add(Dense(400, activation='relu')) 
text_model.add(Dense(200, activation='relu')) 
text_model.add(Dense(100, activation='relu')) 
text_model.add(Dense(5, activation='sigmoid'))

train_feat = tfidf_model.transform(X_trainn)
test_feat = tfidf_model.transform(X_testt)
Y = pd.get_dummies(y_train).values
Yt = pd.get_dummies(y_test).values
# Compile model
text_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
text_model.fit(train_feat, Y, epochs=epoch, batch_size=batch)
# evaluate the model
scores = text_model.evaluate(test_feat, Yt)
print("\n%s: %.2f%%" % (text_model.metrics_names[1], scores[1]*100))


text_model.save(open(os.path.join(cate_dir, '../trained_models/News_Cate.h5')))

path = os.path.join(cate_dir, "data_set/testt.txt")
testdata = open(path, 'r').read()

linessn = testdata.strip()

porter = PorterStemmer()
liness = porter.stem(linessn) 

Yb = tfidf_model.transform([liness])
prediction = text_model.predict(Yb)
print(np.argmax(prediction))

if np.argmax(prediction) == 0:
    print("This is", np.max(prediction)*100, "entertainment news")
elif np.argmax(prediction) == 1:
    print("This is", np.max(prediction)*100, " sport news")
elif np.argmax(prediction) == 2:
    print("This is", np.max(prediction)*100, " politics news")
elif np.argmax(prediction) == 3:
    print("This is", np.max(prediction)*100, " business news")
elif np.argmax(prediction) == 4:
    print("This is", np.max(prediction)*100, " tech news")

print("The accuracy of prediction for DNN is: ", np.max(prediction)*100)
