from flask import Flask
from flask_sqlalchemy import SQLAlchemy


# needed by beanstalk
application = Flask(__name__) # Application instance


# config
# This 'APP_SETTINGS' is an export PATH locally 
application.config.from_envvar('APP_SETTINGS', silent=True)

# Initialize the database
db = SQLAlchemy(application)