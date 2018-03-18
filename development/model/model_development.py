# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:10:25 2018

@author: yiwei
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.externals import joblib



def train_reader(): 
    # Read Train dataset and fill the missing value with single space. 
    train = pd.read_csv('../data/train.csv').fillna(' ')
    return(train)
    

def test_reader(): 
    # Read Test dataset and fill the missing value with single space. 
    test = pd.read_csv('../data/test.csv').fillna(' ')
    #test_text = test['comment_text']
    return(test)


def binary_creater():
    train = train_reader()
    # Create binary response: toxic or not
    # Create the toxic and not for our prediction response
    train['toxic_count'] = [a + b + c + x + y + z for a ,b ,c, x, y, z in 
                        zip(train['toxic'], train['severe_toxic'], train['obscene'],
                           train['threat'], train['insult'], train['identity_hate'])]
    train.loc[train['toxic_count'] > 0, 'is_toxic'] = 1
    train.loc[train['toxic_count'] == 0, 'is_toxic'] = 0
    return(train)


def word_vec():
    # Use TfidfVectorizer to convert a collection of raw documents to a matrix of
    #  TF-IDF features. Term frequency–inverse document frequency (TF-IDF), is a 
    # numerical statistic that is intended to reflect how important a word is to 
    # a document in a collection or corpus.
    all_text = pd.concat([train_reader()['comment_text'], test_reader()['comment_text']]) # Concatenate
    
    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,  # Sublinear scale as the frequency might not be a good indicator
    strip_accents='unicode', # Accents removed
    analyzer='word',
    token_pattern=r'\w{1,}', # cutting words with ' and - in pieces
    ngram_range=(1, 1),
    max_features=500
    )
    word_vectorizer.fit(all_text)
    filename1 = '../data/word_vectorizer.joblib.pkl'
    _ = joblib.dump(word_vectorizer, filename1, compress=9)
    
    char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=3000)
    char_vectorizer.fit(all_text)
    filename2 = '../data/char_vectorizer.joblib.pkl'
    _ = joblib.dump(char_vectorizer, filename2, compress=9)
    
    
    
def model_creater():    
    
    train = binary_creater()
    train_text = train['comment_text']
    filename1 = '../data/word_vectorizer.joblib.pkl'
    filename2 = '../data/char_vectorizer.joblib.pkl'
    word_vectorizer = joblib.load(filename1)
    char_vectorizer = joblib.load(filename2)

    train_word_features = word_vectorizer.transform(train_text)
    train_char_features = char_vectorizer.transform(train_text)
    train_features = hstack([train_char_features, train_word_features])
    
    train_target = train['is_toxic'].astype('int')
    Model = LogisticRegression(solver='sag')
    Model.fit(train_features, train_target)
    filename = '../data/digits_classifier.joblib.pkl'
    _ = joblib.dump(Model, filename, compress=9)

model_creater()

