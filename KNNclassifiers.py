# KNN = K Nearest Neighbours
# It is a classifying but also a regression algorithm
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
#print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values

Y = data[['class']]
#print(X,Y)
X = np.array(X)

#converting the data
#converting x
Le = LabelEncoder()
for i in range(len(X[0])): #X[0]=first row
    X[:,i] = Le.fit_transform(X[:,i]) #fit=learns parameter from the data; transform=apply those learned parameters to transform the data
#print(X)

#converting y
label_mapping = {
    'unaccept':0,
    'accept':1,
    'good':2,
    'vgood':3,
}
Y['class'] = Y['class'].map(label_mapping)
Y = np.array(Y)
#print(Y)

#create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
knn.fit(X_train,Y_train)#to know accuracy and how data performs we need to separate the data into testing data and training data; it needs labels and features
#Basically we compare the o/p of the model (which is trained using training data) to the testing data and gives the accuracy of the model
prediction = knn.predict(X_test)
accuracy=metrics.accuracy_score(Y_test, prediction)
print('predictions:',prediction)
print('accuracy:',accuracy)

a = 1727
print('actual value: ', Y[a]) #actual value at Y index 1727
print('predicted value: ', knn.predict(X)[a]) #predict value at X index 1727

