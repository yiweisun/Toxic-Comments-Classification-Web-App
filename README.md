# Toxic Comments Classification Web App

## Team
Developer: Yiwei(Phyllis) Sun
Project Owner: Anisha Dubhashi
QA: Jerry Chen

## Project Objective 
This repo can be used to produce a toxic comments classification web app. The data preprocessing steps are written with `R` and the app is written with `Python 3`.

## Project Charter

### Vision
Improving online conversations can help platform facilitate conversations. However, the threat of abusive comments prevents people from discussing the matters they care. One aspect of the study is the negative online behaviors like toxic comments. Correctly classifying toxic comment can potentially limit the impact of such comment and improve the conversations participating parties.

### Mission
Create a web app that can classify toxic comments into one of the two categories (toxic, not toxic). Natural language processing techniques will be applied to extract information from comments input by users. 

### Success Criteria
The web app provides an estimated category of the toxic comment based on model trained on the Wikipedia’s talk page edits and have a great performance of the classification. The result dependent on the input comment from users.

## Data
Wikipedia comments that have been labeled by human raters for toxic behavior from Toxic Comment Classification Challenge in Kaggle Competition [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). 
* I used `R` to clean and chunk the raw data (code in `development/model/data_split.R`) 
* I used `Python 3` to do some EDA (code in `development/eda/Project_EDA.ipynb`) and model development (code in `development/model/model_development.py`).

## Pivotal Tracker
[Link to Pivotal Tracker](https://www.pivotaltracker.com/n/projects/2142803)

## Software & Package requirements
Things you need to get it started:
* [conda](https://anaconda.org/): Either Anaconda or Miniconda is fine for this project.
* [git](https://git-scm.com/): You will most likely need version control.

## Web App Set Up

Suggested Steps to set up the app in a AWS EC2 or Linux.

1. Set up EC2 instance and connect:
* Open an SSH client. 
* Locate your private key file 
* Connect to your instance using its Public DNS

2. Update and install git and conda:
    `sudo yum update`
    `sudo yum install git` 
    `wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh bash Anaconda3-5.1.0-Linux-x86_64.sh`

3. Clone this GitHub repository to local. 
    `git clone https://github.com/yiweisun/msia423_project`

4. Create a virtual environment
    `conda create -n msiapp python=3`
    `source activate msiapp`
    `conda config --add channels conda-forge`

5. Install packages
    `conda install flask`
    `conda install flask-sqlalchemy`
    `conda install psycopg2`
    `conda install numpy`
    `conda install pandas`
    `conda install matplotlib`
    `conda install scikit-learn`
    `python pip install -r requirements.txt`
   
6. Run the application
    `python application.py`

