## News Categorizer

This project is concerned with classifying news content into 5 different categories.

### Preprocessing

The raw texts of each news file is preprocessed by removing punctuations, performing
stemming, and lemmatization.

### Feature Extraction

TFIDF is used for extracting features from the proprocessed news data. We set the max length
of the features to 3000. 

### Classification

Deep Neural Network (DNN) is used for performing classification. 

### API

To convert the model into a REST web application, we used Flask and gunicorn as the web application server
