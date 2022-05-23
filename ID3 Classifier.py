# ID3 Classifier

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 03:30:59 2022

@author: Ahmed Nasser
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset = (pd.read_csv("waterQuality1.csv")).dropna()

X = dataset.iloc[:, :8]  # Extract Features
Y = dataset['Potability']  # Set the Label Column

X = ((X - X.mean()) / X.std())  # Normalization step

# Split dataset into training & testing => 60% : 40%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=1, shuffle=True)

# Create Decision-Tree-Classifer object
clf = DecisionTreeClassifier().fit(X_train, y_train)

# Predict the Classifier for Testing dataset
Prdiction = clf.predict(X_test)

# Print Model Accuracy & Error of modelp-testing
print("Test Accuracy: ", metrics.accuracy_score(y_test, Prdiction) * 100, "%")
print("Mean Square Error: ", metrics.mean_squared_error(np.asarray(y_test), Prdiction))
