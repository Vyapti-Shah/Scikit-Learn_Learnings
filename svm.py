#SVM - Support Vector Machine = effective high dimensional spaces, many kernal functions and used in classification and regression
#kernal functions are functions we use to increase dimensions f(x,y) => (x,y,z)
#when the x and y plane have mix data points in some cases it takes the functions as f(x,y) => (x,y,z) adding the z axis to the function

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
#split it in features and labels
x = iris.data
y = iris.target

classes = ['Iris Setosa','Iris Versicolour','Iris Virginica']
print(x.shape)
print(y.shape)

#hours of study vs good/bad grades
#10 different students
#train with 8 students
#predict with the remaining 2
#level of accuracy 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = svm.SVC()
model.fit(x_train, y_train)
print(model)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

predictions = model.predict(x_test)
acc = accuracy_score(y_test, predictions)

print('predictions: ', predictions)
print('actual: ', y_test)
print('accuracy:', acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])
