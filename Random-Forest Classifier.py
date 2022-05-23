# Random-Forest Classifier

# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H0LBA1LS5odBYflMQJW_AN4HwObIJ1Rz
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("waterQuality1.csv")  # substitute the nulls with the mean of thecolumn not the zero
dataset.sort_values("Sulfate", inplace=True)
dataset.drop_duplicates(subset="Sulfate", keep=False, inplace=True)

dataset['Trihalomethanes'] = dataset['Trihalomethanes'].fillna(dataset.groupby(['Potability'])['Trihalomethanes'].transform('mean'))

dataset["Sulfate"].fillna(value=dataset["Sulfate"].mean(), inplace=True)
dataset["Solids"].fillna(value=dataset["Solids"].mean(), inplace=True);
dataset["Conductivity"].fillna(value=dataset["Conductivity"].mean(), inplace=True);
dataset["Organic_carbon"].fillna(value=dataset["Organic_carbon"].mean(), inplace=True);
dataset['ph'] = dataset['ph'].fillna(dataset.groupby(['Potability'])['ph'].transform('mean'))
dataset['Turbidity'] = dataset['Turbidity'].fillna(dataset.groupby(['Potability'])['Turbidity'].transform('mean'))
dataset.corr()

df = pd.DataFrame(dataset)
df = df.drop_duplicates()

x = dataset.iloc[:, :8].values
y = dataset.iloc[:, -1].values
# print(x)
# print(y)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.222, random_state=50, shuffle=True, stratify=y)
# X_train=shuffle(X_train)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB

Classifier = GaussianNB()
Classifier.fit(X_train, y_train)
y_predict = Classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

cm = confusion_matrix(y_predict, y_test)
ac = accuracy_score(y_predict, y_test)
er = mean_squared_error(y_predict, y_test)
print(cm)
print(ac)
print(er)
from sklearn.model_selection import cross_val_score

print(cross_val_score(GaussianNB(), X_train, y_train, cv=10))

clf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
sv = confusion_matrix(y_predict, y_test)
print(sv)
Am = accuracy_score(y_predict, y_test)
print(Am)
er = mean_squared_error(y_predict, y_test)
print(er)

plt.figure(figsize=(12, 8))

heatmap = sns.heatmap(df.corr(), annot=True)

# matplotlib.rcParams["df.ph"]=(20,10)
plt.hist(df.corr(), rwidth=0.8)  # rwidth is for width of the bar
plt.xlabel("feature set ")
plt.ylabel("protability")
