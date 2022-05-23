# SVM-(ver1) Classifier

# -*- coding: utf-8 -*-
"""
Created on Wed May  4 06:55:57 2022

@author: Yossef
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# load dataset
WaterFile = pd.DataFrame(pd.read_csv("waterQuality1.csv"))

X = WaterFile.iloc[:, :8]  # Extract Features
Y = WaterFile['Potability']  # Set the Label Column

# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# learning the statistical parameters for each of the data and transforming
X = scaler.fit_transform(X)

"""
water =WaterFile.groupby("Potability", group_keys = True)
watersampling =  water.apply(lambda X:X.sample(frac = 0.7))
"""
impu = SimpleImputer(missing_values=np.nan, strategy="median")
impu.fit(X)
X = impu.transform(X)

# Split dataset into training & testing => 60% : 40%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=1, shuffle=True)

# Using kernel function

Water = svm.SVC(kernel='sigmoid')

# Train data
Water.fit(X_train, y_train)

# Test data
prediction = Water.predict(X_test)

# Report
accuracy = Water.score(X_train, y_train)
print("train accuracy : ", accuracy * 100, "%")

accuracy = Water.score(X_test, y_test)
print("test accuracy  : ", accuracy * 100, "%")

print("Mean Square Error : ", metrics.mean_squared_error(np.asarray(y_test), prediction))
