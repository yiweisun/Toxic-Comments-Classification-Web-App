from flask import Flask, render_template, flash, request
from webapp import application, db
from webapp.models import model_predictor


@application.route('/', methods=['GET', 'POST'])

def main():
    return render_template('main.html')


@application.route('/result', methods=['POST'])

def result():
	if request.method == 'POST':
		comment = request.form['comment']
		result = model_predictor(comment) # matrix needed
	if result == 0:
		classification = "Not Toxic"
	else:
		classficiation = "Toxic"

		#return render_template('result.html', comment = comment)
	return render_template('result.html', comment=comment, result=classification)

if __name__ == "__main__":  # Script executed directly?
    application.run(debug=True) # Launch built-in web server and run this Flask webapp
