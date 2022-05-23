from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
dataSet = pd.read_csv("water_potability.csv").dropna() # to remove the nan values
## feature extraction
# 1:9
dataFeautres =dataSet.iloc[ :,0:9] ## rows then the cols
#print (" the  feautes  ", dataFeautres)

label=dataSet['Potability'] # our label
# convert the dataSet into  data frame
dataSetDataFrame= pd.DataFrame(dataSet)
#print(" the length of the dtatframe",len(dataSetDataFrame))##2011
#print(dataSetDataFrame)

# drop the duplicated rows

dataSetDataFrame.drop_duplicates(keep='last', inplace=True)

# normalization step :
def getTheNorm ( col):
    return ( ( col-col.min())/(col.max()-col.min()))
## loop on the data fram to get the norm of the cols :
for col in  dataSetDataFrame.columns :
    dataSetDataFrame[col]= getTheNorm(dataSetDataFrame[col])
scalerObject=MinMaxScaler()
scalerObject.fit(dataSetDataFrame)
scaleDataFrame=scalerObject.fit_transform(dataSetDataFrame)
theScaledDataFrame=pd.DataFrame(scaleDataFrame, columns= dataSetDataFrame.columns)
#print( theScaledDataFrame)
# the importanat feautres
theBestFeautres = SelectKBest(score_func=chi2,k=6)
resultOfFit = theBestFeautres.fit (dataFeautres, label )
dfScoresRes=pd.DataFrame(resultOfFit.scores_)
dfcolumns = pd.DataFrame(dataFeautres.columns)
#print (" the dfcolumns ",dfcolumns)

# stratified sampelling
dataFrameInGroups =dataSetDataFrame.groupby('Potability',group_keys=False) # we won't ad the key to the grouped objects
stratifiedSamples =dataFrameInGroups.apply(lambda x: x.sample(frac=0.8))
## spit the data
dataFeautresTrain,dataFeautresTest,labelTrain,labelTest= train_test_split(dataFeautres,label,test_size=0.2, random_state=1)
# the random state, random state = none --> this will shuffle the data and result of different accuracies for each time the function is called
# the ranndom_state = int value -> this will shuffle the data no oF times = the value and the split the data and the result of the split will be fixed  and will be used for each call

## apply the knn classifier
knnClassifier= KNeighborsClassifier(n_neighbors=4) # i got the value of k by try and error
knnClassifier.fit (dataFeautresTrain ,labelTrain)
knnPredictionResult=knnClassifier.predict(dataFeautresTest)
print(" the result  of the knn is ",knnPredictionResult)
##calculate the accuracy and print the error
modelAccuracy=accuracy_score(labelTest,knnPredictionResult)
modelAccuracy *=100
print(" the prediction score is :",(modelAccuracy))

## the mean square error is :
print(" the mean Square error ",metrics.mean_squared_error( np.asarray(labelTest),knnPredictionResult))

