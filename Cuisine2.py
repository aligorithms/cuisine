import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import json
import numpy as np
import scipy as sp
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import dummy
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
from sklearn import tree
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import SVG
from IPython.display import Image
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

import flask
app = flask.Flask(__name__)

    #-------- MODEL GOES HERE -----------#
import pickle

with open('./pickled_recipe_model.pkl', 'rb') as picklefile:
    u = pickle._Unpickler(picklefile)
    #u.encoding = 'latin1'
    PREDICTOR = u.load()

    #-------- ROUTES GO HERE -----------#
@app.route('/page')
def page():
   with open("alispage.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       ing1 = inputs['ing1']
       ing2 = inputs['ing2']
       ing3 = inputs['ing3']
       ing4 = inputs['ing4']
       ing5 = inputs['ing5']
       ing6 = inputs['ing6']
       ing7 = inputs['ing7']
       ing8 = inputs['ing8']

       ing_all = ing1 + " " + ing2 + " " + ing3 + " " + ing4 + " " + ing5 + " " + ing6 + " " + ing7 + " " + ing8

       #item = np.array([pclass, sex, age, fare, sibsp])
       with open('./pickled_recipe_vectorizer.pkl', 'rb') as picklevector:
           v = pickle._Unpickler(picklevector)
           VECTORIZER = v.load()
           X_test_dtm = VECTORIZER.transform([ing_all])


       item = X_test_dtm
       score = PREDICTOR.predict_proba(item)
       results = {'chinese': score[0,0], 'japanese': score[0,1], 'korean': score[0,2], 'thai': score[0,3], 'vietnamese': score[0,4]}
       return flask.jsonify(results)

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(HOST, PORT)
