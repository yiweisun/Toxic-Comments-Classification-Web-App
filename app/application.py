from models import model_predictor
from flask import Flask, render_template, request, session, g, redirect, url_for, abort, \
    render_template, flash
import os

from sklearn.externals import joblib

application = Flask(__name__)
application.config.from_object(__name__) # Load config


    
filename1 = '../development/data/word_vectorizer.joblib.pkl'
filename2 = '../development/data/char_vectorizer.joblib.pkl'
word_vectorizer = joblib.load(filename1)
char_vectorizer = joblib.load(filename2)
filename = '../development/data/digits_classifier.joblib.pkl'
Model = joblib.load(open(filename, 'rb'))

@application.route('/', methods=['GET', 'POST'])

def main():
    return render_template('main.html')


@application.route('/result', methods=['POST'])

def result():
	if request.method == 'POST':
		comment = request.form['comment']
		result = model_predictor(comment,word_vectorizer,char_vectorizer,Model) # matrix needed
	if result == 0:
		classification = "Not Toxic"
	else:
		classification = "Toxic"

		#return render_template('result.html', comment = comment)
	return render_template('result.html', comment=comment, result=classification)

if __name__ == "__main__":  # Script executed directly?
    application.run(host = "0.0.0.0:80",debug=True) # Launch built-in web server and run this Flask webapp

# application.run(host = "0.0.0.0", debug=True)