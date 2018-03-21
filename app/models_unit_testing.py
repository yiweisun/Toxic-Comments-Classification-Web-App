"""

This is a Unit Testing file for models.py. 

Author: Yiwei Sun

"""

from sklearn.externals import joblib

def test_model_predictor():
    """Test model_predictor function."""
    
    # str input
    test = 'Atheism is full of bias shit'

    # Word and character vectorizer and the logistic model inputs
    filename1 = '../development/data/word_vectorizer.joblib.pkl'
    filename2 = '../development/data/char_vectorizer.joblib.pkl'
    word_vectorizer = joblib.load(filename1)
    char_vectorizer = joblib.load(filename2)
    filename = '../development/data/digits_classifier.joblib.pkl'
    Model = joblib.load(open(filename, 'rb'))
    
    # Expected output
    expected_int = 1
    
    try:
        # Check type
        assert isinstance(test, str)
        
        # Check function output
        assert (expected_int == model_predictor(test, word_vectorizer,\
                                            char_vectorizer,Model))
        print('model_predictor() function test passed!')
        
    except:
        print('model_predictor() function test failed!')
    
    