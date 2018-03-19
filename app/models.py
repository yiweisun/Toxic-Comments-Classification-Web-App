# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:10:25 2018

@author: yiwei
"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
#model_development.path.append('../development/model')
#from modelmodel_development import binary_creater



def model_predictor(new,word_vectorizer,char_vectorizer,Model):
    test = {'comment_text': [new]}
    test = pd.DataFrame(data=test)

    
    test = {'comment_text': [new]}
    test = pd.DataFrame(data=test)
    test_text = test['comment_text']
    test_word_features = word_vectorizer.transform(test_text)
    test_char_features = char_vectorizer.transform(test_text)
    test_feature = hstack([test_char_features, test_word_features])

    
    
    pred = Model.predict(test_feature)
    return(pred)



