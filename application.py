
"""

This is the flask application page for the toxic comments classification webapp.

Author: Yiwei Sun

"""

from app.models import model_predictor
from flask import Flask, render_template, request, session, g, redirect, url_for, abort, \
render_template, flash
import os
from sklearn.externals import joblib
import logging


application = Flask(__name__)
application.config.from_object(__name__) # Load config


# Load pickle file for word and character vectorizer    
filename1 = 'development/data/word_vectorizer.joblib.pkl'
filename2 = 'development/data/char_vectorizer.joblib.pkl'
word_vectorizer = joblib.load(filename1)
char_vectorizer = joblib.load(filename2)
# Load pickle file for logistic regression 
filename = 'development/data/digits_classifier.joblib.pkl'
Model = joblib.load(open(filename, 'rb'))


@application.route('/', methods=['GET', 'POST'])

def main():
    """Home page of the webapp

    Args:

        Null



    Returns:

        flask-obj: rendered html page

    """
  
    logger.info("Go to the main page.")
    return render_template('main.html')


@application.route('/result', methods=['POST'])

def result():
    """Result page of webapp

    Args:

        Null

    Returns:

        flask-obj: rendered html page

    """

    if request.method == 'POST':
         comment = request.form['comment']
         result = model_predictor(comment,word_vectorizer,char_vectorizer,Model) # matrix needed
         logger.info("Classification has been completed.")
      
    if result == 0:
         classification = "Not Toxic"
    else:
         classification = "Toxic"

    return render_template('result.html', comment=comment, result=classification)

if __name__ == "__main__":  
     # logger initialization
    logging.basicConfig(filename='application.log', level=logging.DEBUG)
    logger = logging.getLogger(__name__) 
     # Launch built-in web server and run this Flask webapp
    application.run(host = "0.0.0.0",debug=True) 

