# Logistic Classifier

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

dataFile = (pd.read_csv("waterQuality1.csv")).dropna()  # Reading data

X = dataFile.iloc[:, :8]  # Extract Features
Y = dataFile['Potability']  # Set the Label Column

X = ((X - X.mean()) / X.std())  # Normalization step

# Split dataset into training & testing => 60% : 40%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=1, shuffle=True)

clf = LogisticRegression().fit(X_train, y_train)

# Test data
Prediction = clf.predict(X_test)
confusion_matrix(y_test, Prediction)

# Print Model Accuracy & Error of modelp-testing
print("Test Accuracy: ", metrics.accuracy_score(y_test, Prediction) * 100, "%")
print("Mean Square Error: ", metrics.mean_squared_error(np.asarray(y_test), Prediction))

print('\n\nReport: ', classification_report(y_test, Prediction), sep='\n')
