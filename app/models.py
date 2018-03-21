"""

This is the classification file for the toxic comment webapp

Author: Yiwei Sun

"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def model_predictor(new,word_vectorizer,char_vectorizer,Model):
    """Predict the class for the input by customer

    Args:
        new(str): A String of customer input. 
        word_vectorizer(pkl): A pickle file saving word vectorizer from TfidfVectorizer()
        char_vectorizer(pkl): A pickle file saving character vectorizer from TfidfVectorizer()
        Model(pkl): A pickle file saving the logistic model used to classify

    Returns:
        str: the predicted class for the imput string

    """

    # Save the input string in pandas dataframe
    test = {'comment_text': [new]}
    test = pd.DataFrame(data=test)
    test_text = test['comment_text']

    # Vectorize the word and character for the new input
    test_word_features = word_vectorizer.transform(test_text)
    test_char_features = char_vectorizer.transform(test_text)
    test_feature = hstack([test_char_features, test_word_features])

    # Classify the input by the logistic regression input
    pred = Model.predict(test_feature)
    return(pred)



