"""This script is a sample submission for the Leaders Prize competition.
It reads in a dataset and creates a sample predictions file.
"""

import json
import os
from random import randint


#####################
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import itertools
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
# from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
#####################
#####################

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = '/usr/local/dataset/metadata.json'
ARTICLES_FILEPATH = '/usr/local/dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = '/usr/local/predictions.txt'

#####################
def appendArticles(articleList, basePath):
    contents = ''
    for idx in articleList:
        with open(os.path.join(basePath, '%d.txt' % idx), 'r') as f:
            contents = f.read()+";"+contents
    return contents


def assignLength(row, colName):
    return len(row[colName])
print('########################')
basePath = "/usr/local"
#basePath = os.path.dirname(os.path.abspath("train.json"))
txtPath = os.path.join(basePath, "train_articles")
jsonPath = os.path.join(basePath, "train.json")
print(jsonPath)
# 0:false, 1:partly true, 2:true
claim = pd.read_json(open(jsonPath, "r", encoding="utf8"))
claim['articleText'] = claim.apply(lambda row: \
                                   appendArticles(row['related_articles'], txtPath) ,axis=1)
claim['articleLength'] = claim.apply(lambda row: \
                                     assignLength(row, 'articleText'), axis=1)
y_train = claim['label']
X_train = claim.drop("label", axis=1)

# Initialize the `tfidf_vectorizer`
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train['articleText'])

#####################
#####################

# Read in the metadata file.
#with open(METADATA_FILEPATH, 'r') as f:
#    claims = json.load(f)
#
## Inspect the first claim.
#claim = claims[0]
#print('Claim:', claim['claim'])
#print('Speaker:', claim['claimant'])
#print('Date:', claim['date'])
#print('Related Article Ids:', claim['related_articles'])


#####################
X_test = pd.read_json(open(METADATA_FILEPATH, "r", encoding="utf8"))
X_test['articleText'] = X_test.apply(lambda row: \
                                   appendArticles(row['related_articles'], ARTICLES_FILEPATH) ,axis=1)
X_test['articleLength'] = X_test.apply(lambda row: \
                                     assignLength(row, 'articleText'), axis=1)
if "lable" in X_test:
    X_test = X_test.drop("label", axis=1)

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test['articleText'])

#####################
#####################

## Print the first evidence article.
#idx = claim['related_articles'][0]
#print('First evidence article id:', idx)
#with open(os.path.join(ARTICLES_FILEPATH, '%d.txt' % idx), 'r') as f:
#    print(f.read())


#####################
clf = MultinomialNB(alpha=0.1)
nb_classifier.fit(tfidf_train, y_train)
pred = nb_classifier.predict(tfidf_test)

with open(PREDICTIONS_FILEPATH, 'w') as f:
    finalResults = zip(X_test['id'], pred)
    for eachPair in finalResults:
        f.write('%d,%d\n' % (eachPair[0], eachPair[1]))
print('Finished writing predictions.')


#####################
#####################

# Create a predictions file.
#print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
#with open(PREDICTIONS_FILEPATH, 'w') as f:
#    for claim in claims:
#        f.write('%d,%d\n' % (claim['id'], randint(0, 2)) )
#print('Finished writing predictions.')
