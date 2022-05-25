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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score

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

le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.222, random_state=50, shuffle=True, stratify=y)
# X_train=shuffle(X_train)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Classifier = GaussianNB()
Classifier.fit(X_train, y_train)
y_predict = Classifier.predict(X_test)

cm = confusion_matrix(y_predict, y_test)
ac = accuracy_score(y_predict, y_test)
er = mean_squared_error(y_predict, y_test)

print(
    "GaussianNB Classifier", '\n\t',
    "Confusion Matrix", '\n\t\t', cm[0, :], '\n\t\t', cm[1, :], '\n\t',
    "Accuracy = ", ac*100, '\n\t',
    "Mean Squared Error = ", er, '\n'
)

print("cross_val_score GaussianNB\n", cross_val_score(GaussianNB(), X_train, y_train, cv=10), '\n')

clf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

sv = confusion_matrix(y_predict, y_test)
Am = accuracy_score(y_predict, y_test)
er = mean_squared_error(y_predict, y_test)

print(
    "RandomForest Classifier", '\n\t',
    "Confusion Matrix", '\n\t\t', sv[0, :], '\n\t\t', sv[1, :], '\n\t',
    "Accuracy = ", Am*100, '\n\t',
    "Mean Squared Error = ", er, '\n'
)

plt.figure(figsize=(12, 8))

heatmap = sns.heatmap(df.corr(), annot=True)

# matplotlib.rcParams["df.ph"]=(20,10)
plt.hist(df.corr(), rwidth=0.8)  # rwidth is for width of the bar
plt.xlabel("Features")
plt.ylabel("Potability")
plt.show()
