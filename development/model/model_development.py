"""

This is the model development file for the toxic comment webapp.

Author: Yiwei Sun

"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def train_sub_reader(): 
    """Read train set data 

        Args:
            Null

        Returns:
            df: the readed train set data in panda dataframe

    """

    # Read Train dataset and fill the missing value with single space. 
    train = pd.read_csv('../data/train_sub.csv').fillna(' ')
    return(train)
    
def binary_creater():
    """Create response variable for logistic regression 

        Args:
            Null

        Returns:
            df: the train set data with binary indicator added

    """

    train = train_sub_reader()
    # Create binary response: toxic or not
    # Create the toxic and not for our prediction response
    train['toxic_count'] = [a + b + c + x + y + z for a ,b ,c, x, y, z in 
                        zip(train['toxic'], train['severe_toxic'], train['obscene'],
                           train['threat'], train['insult'], train['identity_hate'])]
    train.loc[train['toxic_count'] > 0, 'is_toxic'] = 1
    train.loc[train['toxic_count'] == 0, 'is_toxic'] = 0
    return(train)


def word_vec():
    """Train word and characters vectorizer from the train set text and save in pickle files"""

    # Use TfidfVectorizer to convert a collection of raw documents to a matrix of
    #  TF-IDF features. Term frequency–inverse document frequency (TF-IDF), is a 
    # numerical statistic that is intended to reflect how important a word is to 
    # a document in a collection or corpus.
    train_sub_text = train_sub_reader()['comment_text']
    
    # Train word vectorizer
    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,  # Sublinear scale as the frequency might not be a good indicator
    strip_accents='unicode', # Accents removed
    analyzer='word',
    token_pattern=r'\w{1,}', # cutting words with ' and - in pieces
    ngram_range=(1, 1),
    max_features=500
    )
    word_vectorizer.fit(train_sub_text)
    filename1 = '../data/word_vectorizer.joblib.pkl'
    _ = joblib.dump(word_vectorizer, filename1, compress=9)
    
    
    # Train character vectorizer
    char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=3000)
    char_vectorizer.fit(train_sub_text)
    filename2 = '../data/char_vectorizer.joblib.pkl'
    _ = joblib.dump(char_vectorizer, filename2, compress=9)
    


def model_creater():    
    """Fit the logistic regression based on word and character features through vectorization"""

    train = binary_creater()
    train_text = train['comment_text']
    filename1 = '../data/word_vectorizer.joblib.pkl'
    filename2 = '../data/char_vectorizer.joblib.pkl'
    word_vectorizer = joblib.load(filename1)
    char_vectorizer = joblib.load(filename2)

    # Get features of word and characters in the train set
    train_word_features = word_vectorizer.transform(train_text)
    train_char_features = char_vectorizer.transform(train_text)
    train_features = hstack([train_char_features, train_word_features])
    train_target = train['is_toxic'].astype('int')

    # Generate the logistic regression by fitting word and characters features
    Model = LogisticRegression(solver='sag')
    Model.fit(train_features, train_target)

    # Save the logistic regression in the pickle file
    filename = '../data/digits_classifier.joblib.pkl'
    _ = joblib.dump(Model, filename, compress=9)


def cv_tester():
    """Use Cross Validation test the performance of our logistic regression"""
    train = binary_creater()
    train_text = train['comment_text']
    filename = '../data/digits_classifier.joblib.pkl'
    filename1 = '../data/word_vectorizer.joblib.pkl'
    filename2 = '../data/char_vectorizer.joblib.pkl'
    word_vectorizer = joblib.load(filename1)
    char_vectorizer = joblib.load(filename2)

    train_word_features = word_vectorizer.transform(train_text)
    train_char_features = char_vectorizer.transform(train_text)
    train_features = hstack([train_char_features, train_word_features])
    train_target = train['is_toxic'].astype('int')
    
    Model = joblib.load(filename)
    cv_score = np.mean(cross_val_score(Model, train_features, 
                                       train_target, cv=5, scoring='accuracy'))
    print(cv_score)
    
def tester():
     """Use test set data test the performance of our logistic regression"""
    test = test_sub_reader()
    test_text = test['comment_text']
    filename = '../data/digits_classifier.joblib.pkl'
    filename1 = '../data/word_vectorizer.joblib.pkl'
    filename2 = '../data/char_vectorizer.joblib.pkl'
    word_vectorizer = joblib.load(filename1)
    char_vectorizer = joblib.load(filename2)

    test_word_features = word_vectorizer.transform(test_text)
    test_char_features = char_vectorizer.transform(test_text)
    test_features = hstack([test_char_features, test_word_features])
    test_target = test['binary'].astype('int')
    
    Model = joblib.load(filename)
    accuracy = Model.score(test_features, test_target)
    print(accuracy)
    
