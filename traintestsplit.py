from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split  

iris = datasets.load_iris()

#split it in features and labels
X = iris.data
Y = iris.target
print(X,Y)

print(X.shape)
print(Y.shape)

#hours of study vs good/bad grades
#10 different  students
#train with 8 students
#predict with the remaining 2
#level of accuracy

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2) #test_size=0.2 means 20% of the data (30 samples) will be taken for testing and 80% (120 samples) taken for training ie.
print(X_train.shape) # (120, 4) → 120 samples for training.
print(X_test.shape) # (30, 4) → 30 samples for testing.
print(Y_train.shape) # (120,) → 120 labels for training.
print(Y_test.shape) # (30,) → 30 labels for testing.
#training data is larger than testing data but the testing data is not too low because if it is too low then the accuracy will be less 

