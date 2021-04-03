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

## Getting Started

The following steps will guide you through setting up the project

First, clone the repo

git clone https://github.com/olahsymbo/NewsCategorizer.git

### Install and configure virtual environment

Goto project directory and setup virtual environment:

```
cd NewsCategorizer
pip3 install virtualenv
virtualenv news
source news/bin/activate
```

To ensure this app functions properly, install the dependencies in the requirements.txt Simply run:

`pip install -r requirements.txt`

### Create a Postgres DB

Install postgresql 12.2 (incase it's not installed already).

Create a new postgres database named `news` using:

CREATEDB news;

inside the db shell, create username and password for the database using:

CREATE USER newsdev with encrypted password 'newsdev';

### Run the NewsCategorizer

Launch the flask web server using:

` python3 api/news_api.py`

The base url is:

`http://127.0.0.1:5000/`
