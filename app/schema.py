from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# flask application
application = Flask(__name__)

# config
application.config.from_pyfile('config.py', silent = True)

# Initialize the database
db = SQLAlchemy(application)

class Web(db.Model):
	index = db.Column(db.Integer, primary_key = True)
	comment = db.Column(db.String(40), nullable = False)
	classification = db.Column(db.String(40), nullable = False)

	def __repr__(self):
		return '<URL %r>' % self.url