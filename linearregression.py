from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

iris = datasets.load_iris()

#featres / labels
x = iris.data
y = iris.target
print("X:")
print(x)
print(x.shape)
print("Y:")
print(y)
print(y.shape)
#columns are all features and rows are all instances

#algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(x.T[0],y) #0th feature of x
plt.show()

plt.scatter(x.T[3],y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#train 
model = l_reg.fit(x_train, y_train)
predictions = model.predict(x_test)

print("Prediction: ", predictions)
print("R^2 value: ", l_reg.score(x,y))
print("coef: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)
