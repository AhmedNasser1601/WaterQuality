import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

dataset = (pd.read_csv("water_potability.csv")).dropna()
dataset = dataset.drop_duplicates()#Drop Duplicates
X = dataset.iloc[:, :5]   #Extract Features
Y = dataset['Potability']   #Set the Label Column

X = ((X - X.mean()) / X.std())   #Normalization step

#Split dataset into training & testing => 80% : 20%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.20, random_state= 1, shuffle= True)
#create and train the K Nearest Neighbor model with the training set
classifier = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski', p= 2)
classifier.fit(X_train, y_train)

#Comparing true and predicted value
y_pred = classifier.predict(X_test)

#evaluate our model using the confusion matrix and accuracy score by comparing the predicted and actual test values
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print("confusion matrix \n",cm)
print("\nAccuracy =",ac*100)
print("Error =" ,1-ac)