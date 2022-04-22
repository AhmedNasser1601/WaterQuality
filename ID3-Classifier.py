# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 03:30:59 2022

@author: Ahmed Nasser
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

Waterfile = (pd.read_csv("water_potability.csv")).dropna()

X = Waterfile.iloc[0:3275,: ]     #Features
Y = Waterfile['Potability']       #Label
X = ((X - X.mean()) / X.std())    #Normalization

# Split dataset into training set and test set => 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1, shuffle = True)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier().fit(X_train, y_train)

#Predict the response for test dataset
Prdiction = clf.predict(X_test)

# Model Accuracy, Correctness of the model
print("Test Accuracy: ", metrics.accuracy_score(y_test, Prdiction) * 100, "%") #Test Prediction
print("Mean Square Error: ", metrics.mean_squared_error(np.asarray(y_test), Prdiction)) #print(Waterfile)