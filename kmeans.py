from sklearn.datasets import load_breast_cancer
from sklearn.cluster import  KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd 

bc = load_breast_cancer()
#print(bc)

x = scale(bc.data)
print(x)

y = bc.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=6)
model = KMeans(n_clusters=2,random_state=0)
model.fit(x_train) #training the model
predictions = model.predict(x_test)
labels = model.labels_
print('Labels',labels)
print("Predictions: ",predictions)
print('Accuracy: ',accuracy_score(y_test,predictions))
print('Actual: ',y_test)
#cluster 0 represents label 0

print(pd.crosstab(y_train,labels))

#Another way for finding actual print(pd.crosstab(y_train,labels))
from sklearn import metrics
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
           % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
bench_k_means(model, "1", x)
print(pd.crosstab(y_train, labels))