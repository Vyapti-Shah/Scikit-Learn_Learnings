import numpy as np
import pandas as pd
from sklearn import neighbors, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

data = pd.read_csv('car.data')
print(data.head())

X = data[['buying', 'maint', 'safety']].values #features
y = data[['class']] #labels
X = np.array(X)
print(X)
#converting data
#X
Le = LabelEncoder() #LabelEncoder() is used to convert categorical features into numerical values.
for i in range(len(X[0])): #It loops over the columns (i.e., 'buying', 'maint', 'safety') and transforms them into numbers.
    X[:, i] = Le.fit_transform(X[:, i])
print(X)
#y
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping) #Maps car class labels (unacc, acc, good, vgood) to numeric values.
y = np.array(y)

#create model
print(X.shape) #Prints the shape of feature data (n_samples, n_features)
print(y.shape) #Prints the shape of label data (n_samples)

knn = svm.SVC() #SVM(Support Vector Classifier)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2) 
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions:", prediction)
print("accuracy: ", accuracy)