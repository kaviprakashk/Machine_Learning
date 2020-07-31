from sklearn.datasets import load_breast_cancer
breast_cancer=load_breast_cancer()
print(breast_cancer)

print(breast_cancer.keys())
print(breast_cancer.data)
print(breast_cancer.data.shape)
breast_cancer.target.shape
print(breast_cancer.DESCR)

import pandas as pd
bc=pd.DataFrame(breast_cancer.data)
print(bc)

breast_cancer.feature_names
bc.columns=breast_cancer.feature_names
print(bc)

bc.describe()

bc['result']=breast_cancer.target
print(bc)

bc.shape
x=bc.drop('result',axis='columns')
y=bc.result
print(x.shape)
print(y.shape)
print(x)
print(y)

import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3,random_state=7)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
Y_pred = lr.predict(X_test)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
plt.figure(figsize=(4,3))
plt.scatter(Y_test,Y_pred)
plt.xlabel("Result")
plt.ylabel("Predicted Result")
plt.title("Result vs Predicted Result")

mse=sklearn.metrics.mean_squared_error(Y_test,Y_pred)
mse

for index, feature_name in enumerate(breast_cancer.feature_names):
  plt.figure(figsize=(4,3))
  plt.scatter(breast_cancer.data[:, index], breast_cancer.target)
  plt.title(feature_name,size=30)
  plt.ylabel("Result",size=15)
  plt.xlabel(feature_name,size=15)
  plt.tight_layout()

from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(X_train,Y_train)

predicted=gbr.predict(X_test)
expected=Y_test
plt.figure(figsize=(4,3))
plt.scatter(predicted,expected)
plt.plot([0, 1.5], [0, 1.5], 'r')
plt.axis("tight")
plt.xlabel("True Result")
plt.ylabel("Predicted Result")
plt.title("True Result vs Predicted Result")
plt.tight_layout()

from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(X_train,Y_train)

mse=sklearn.metrics.mean_squared_error(predicted,expected)
mse

plt.figure(figsize=(4, 3))
plt.hist(breast_cancer.target)
plt.tight_layout()
