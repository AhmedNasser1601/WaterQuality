# Gradient-Boosting Classifier

# -*- coding: utf-8 -*-
"""Gradient Boosting Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yxUB0L6Z7SNvRJPtym1qB0BipsB7z4S5
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# loading csv file
file_path = "waterQuality1.csv"
df = pd.read_csv(file_path)

# preprocessing
# dropping null_valued cells
df.drop_duplicates(keep='first', inplace=True)
# replacing outliers by nulls
for i in ['Chloramines']:
    q75, q25 = np.percentile(df.loc[:, i], [75, 25])
    intr_qr = q75 - q25
    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    df.loc[df[i] < lower, i] = np.nan
    df.loc[df[i] > upper, i] = np.nan

for i in ["Hardness"]:
    q75, q25 = np.percentile(df.loc[:, i], [75, 25])
    intr_qr = q75 - q25

    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    df.loc[df[i] < lower, i] = np.nan
    df.loc[df[i] > upper, i] = np.nan

for i in ["Conductivity"]:
    q75, q25 = np.percentile(df.loc[:, i], [75, 25])
    intr_qr = q75 - q25

    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    df.loc[df[i] < lower, i] = np.nan
    df.loc[df[i] > upper, i] = np.nan

for i in ["Organic_carbon"]:
    q75, q25 = np.percentile(df.loc[:, i], [75, 25])
    intr_qr = q75 - q25

    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    df.loc[df[i] < lower, i] = np.nan
    df.loc[df[i] > upper, i] = np.nan

for x in ["Trihalomethanes"]:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df.loc[df[x] < min, x] = np.nan
    df.loc[df[x] > max, x] = np.nan

for x in ["Turbidity"]:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df.loc[df[x] < min, x] = np.nan
    df.loc[df[x] > max, x] = np.nan

# repacing the null values by the mean of eavery feature
df["ph"].fillna(value=df["ph"].mean(), inplace=True)
df["Hardness"].fillna(value=df["Hardness"].mean(), inplace=True)
df["Chloramines"].fillna(value=df["Chloramines"].mean(), inplace=True)
df["Conductivity"].fillna(value=df["Conductivity"].mean(), inplace=True)
df["Organic_carbon"].fillna(value=df["Organic_carbon"].mean(), inplace=True)
df["Turbidity"].fillna(value=df["Turbidity"].mean(), inplace=True)
df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].mean(), inplace=True)
df["Sulfate"].fillna(value=df["Sulfate"].mean(), inplace=True)

# spliting the dataset
X = df.drop('Potability', axis=1)
Y = df.Potability

# scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.222, random_state=42)

# Now we can try setting different learning rates, so that we can compare the performance of the classifier's performance at different learning rates.
learningRate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for lr in learningRate_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=lr, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print(
        "Learning rate: ", lr, '\t',
        "Training Accuracy: {0:.3f}".format(gb_clf.score(X_train, y_train)), '\t',
        "Validation Accuracy: {0:.3f}".format(gb_clf.score(X_test, y_test) * 100), '%\n'
    )
