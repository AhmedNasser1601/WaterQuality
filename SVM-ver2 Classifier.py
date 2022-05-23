# SVM-(ver2) Classifier

# -*- coding: utf-8 -*-
"""SVM #2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eTH2M9oryt6TPoUDFOYMKMBvLxHQDHPs
"""

import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split

# loading csv file
file_path = "waterQuality1.csv"
dataset = pd.read_csv(file_path)
dataset.head()

# preprocessing

dataset.drop(['ph', 'Sulfate'], axis=1, inplace=True)
dataset.dropna(inplace=True)

# remove duplicates
dataset.drop_duplicates(keep='first', inplace=True)

# Model
X = dataset.drop('Potability', axis=1)
Y = dataset.Potability

x_scaled = preprocessing.scale(X)
# splitting the data
X_train, X_test, y_train, y_test = train_test_split(x_scaled, Y, test_size=0.22, random_state=42)

# create the svm classifier
clf = svm.SVC(kernel='linear')

# train the model
clf.fit(X_train, y_train)  #

# predict the response for dataset
y_pred = clf.predict(X_test)

# accurracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)
