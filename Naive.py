from audioop import avg

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error
from  sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.impute import SimpleImputer

#Read Data
dataset = pd.read_csv("water_potability.csv")

x = dataset.iloc[:,:8].values   
y = dataset.iloc[:,-1].values
#print(x)
#print(y)

#Data Preproccessing
le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])

impu = SimpleImputer(missing_values= np.nan , strategy="mean")
impu.fit(x)
X = impu.transform(x)

#Splitting Data
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size= 0.2, random_state= 1, shuffle= True, stratify= y)

sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train)
#print(X_test)

Classifier = GaussianNB()   

#Train data
Classifier.fit(X_train,y_train)

#Test data
y_predict = Classifier.predict(X_test)

#Calculate accuracy
cm = confusion_matrix(y_predict,y_test)
ac = accuracy_score(y_predict,y_test) *100
er = mean_squared_error(y_predict,y_test)
print("Confusion matrix \n",cm)
print("Accuracy ---->", ac)
print("Mean square error ----->",er)

print("\n Cross validation \n",cross_val_score(GaussianNB(),X_train,y_train,cv=5))
