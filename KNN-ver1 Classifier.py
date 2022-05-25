# KNN-(ver1) Classifier

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataSet = pd.read_csv("waterQuality1.csv").dropna()  # to remove the nan values
## feature extraction
# 1:9
dataFeautres = dataSet.iloc[:, 0:9]  ## rows then the cols
# print (" the  feautes  ", dataFeautres)

label = dataSet['Potability']  # our label
# convert the dataSet into  data frame
dataSetDataFrame = pd.DataFrame(dataSet)
# print(" the length of the dtatframe",len(dataSetDataFrame))##2011
# print(dataSetDataFrame)

# drop the duplicated rows

dataSetDataFrame.drop_duplicates(keep='last', inplace=True)
# replacing outliers by nulls
for i in ['Chloramines']:
    q75, q25 = np.percentile(dataSetDataFrame.loc[:, i], [75, 25])
    intr_qr = q75 - q25
    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    dataSetDataFrame.loc[dataSetDataFrame[i] < lower, i] = np.nan
    dataSetDataFrame.loc[dataSetDataFrame[i] > upper, i] = np.nan

for i in ["Hardness"]:
    q75, q25 = np.percentile(dataSetDataFrame.loc[:, i], [75, 25])
    intr_qr = q75 - q25

    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    dataSetDataFrame.loc[dataSetDataFrame[i] < lower, i] = np.nan
    dataSetDataFrame.loc[dataSetDataFrame[i] > upper, i] = np.nan

for i in ["Conductivity"]:
    q75, q25 = np.percentile(dataSetDataFrame.loc[:, i], [75, 25])
    intr_qr = q75 - q25

    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    dataSetDataFrame.loc[dataSetDataFrame[i] < lower, i] = np.nan
    dataSetDataFrame.loc[dataSetDataFrame[i] > upper, i] = np.nan

for i in ["Organic_carbon"]:
    q75, q25 = np.percentile(dataSetDataFrame.loc[:, i], [75, 25])
    intr_qr = q75 - q25

    upper = q75 + (1.5 * intr_qr)
    lower = q25 - (1.5 * intr_qr)

    dataSetDataFrame.loc[dataSetDataFrame[i] < lower, i] = np.nan
    dataSetDataFrame.loc[dataSetDataFrame[i] > upper, i] = np.nan

for x in ["Trihalomethanes"]:
    q75, q25 = np.percentile(dataSetDataFrame.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    dataSetDataFrame.loc[dataSetDataFrame[x] < min, x] = np.nan
    dataSetDataFrame.loc[dataSetDataFrame[x] > max, x] = np.nan

for x in ["Turbidity"]:
    q75, q25 = np.percentile(dataSetDataFrame.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    dataSetDataFrame.loc[dataSetDataFrame[x] < min, x] = np.nan
    dataSetDataFrame.loc[dataSetDataFrame[x] > max, x] = np.nan

# repacing the null values by the mean of eavery feature
dataSetDataFrame["ph"].fillna(value=dataSetDataFrame["ph"].mean(), inplace=True)
dataSetDataFrame["Hardness"].fillna(value=dataSetDataFrame["Hardness"].mean(), inplace=True)
dataSetDataFrame["Chloramines"].fillna(value=dataSetDataFrame["Chloramines"].mean(), inplace=True)
dataSetDataFrame["Conductivity"].fillna(value=dataSetDataFrame["Conductivity"].mean(), inplace=True)
dataSetDataFrame["Organic_carbon"].fillna(value=dataSetDataFrame["Organic_carbon"].mean(), inplace=True)
dataSetDataFrame["Turbidity"].fillna(value=dataSetDataFrame["Turbidity"].mean(), inplace=True)
dataSetDataFrame['Trihalomethanes'].fillna(value=dataSetDataFrame['Trihalomethanes'].mean(), inplace=True)
dataSetDataFrame["Sulfate"].fillna(value=dataSetDataFrame["Sulfate"].mean(), inplace=True)


# normalization step :
def getTheNorm(col):
    return ((col - col.min()) / (col.max() - col.min()))


## loop on the data fram to get the norm of the cols :
for col in dataSetDataFrame.columns:
    dataSetDataFrame[col] = getTheNorm(dataSetDataFrame[col])
## another way of the normalization using sklearn
'''scalerObject=MinMaxScaler()
scalerObject.fit(dataSetDataFrame) # passes the dataframe into the method 
scaleDataFrame=scalerObject.fit_transform(dataSetDataFrame)# create a sacled matrix ( normalized matrix ya3ne )
theScaledDataFrame=pd.DataFrame(scaleDataFrame, columns= dataSetDataFrame.columns)# recreate the dataframe using DataFrame calss of pandas 
'''
# print( theScaledDataFrame)
# the importanat feautres

theBestFeautres = SelectKBest(score_func=chi2, k=7)
resultOfFit = theBestFeautres.fit(dataFeautres, label)
theMask = theBestFeautres.get_support()
thenewFeautres = dataFeautres.columns[theMask]
# theScaledDataFrame=pd.DataFrame(scaleDataFrame, columns= dataSetDataFrame.columns)
print("  the new feautres are \n", thenewFeautres)
dataSetDataFrame = pd.DataFrame(dataSetDataFrame, columns=thenewFeautres)
dataSetDataFrame['Potability'] = label

# print (" the data frame after the modifications\n ", dataSetDataFrame)


# stratified sampelling
dataFrameInGroups = dataSetDataFrame.groupby('Potability',
                                             group_keys=False)  # we won't ad the key to the grouped objects
stratifiedSamples = dataFrameInGroups.apply(lambda x: x.sample(frac=0.8))
## spit the data
dataFeautresTrain, dataFeautresTest, labelTrain, labelTest = train_test_split(dataFeautres, label, test_size=0.2,
                                                                              random_state=1)
# the random state, random state = none --> this will shuffle the data and result of different accuracies for each time the function is called
# the ranndom_state = int value -> this will shuffle the data no oF times = the value and the split the data and the result of the split will be fixed  and will be used for each call

## apply the knn classifier
knnClassifier = KNeighborsClassifier(n_neighbors=4)  # i got the value of k by try and error
knnClassifier.fit(dataFeautresTrain, labelTrain)
knnPredictionResult = knnClassifier.predict(dataFeautresTest)
print(" the result  of the knn is ", knnPredictionResult)
##calculate the accuracy and print the error
modelAccuracy = accuracy_score(labelTest, knnPredictionResult)
modelAccuracy *= 100
print(" the prediction score is :", (modelAccuracy))

## the mean square error is :
print(" the mean Square error ", metrics.mean_squared_error(np.asarray(labelTest), knnPredictionResult))
