from sklearn.datasets import load_boston
boston=load_boston()
print(boston)

print(boston.keys())
print(boston.target.shape)
print(boston.DESCR)
print(boston.data)
print(boston.target)

import pandas as pd
bos=pd.DataFrame(boston.data)
print(bos)

print(boston.feature_names)
bos.describe()

bos['price']=boston.target
bos.shape
print(bos)

x=bos.drop('price',axis=1)
y=bos['price']
print(x.shape)
print(y.shape)
print(bos.shape)

import warnings
import sklearn
from sklearn.model_selection import train_test_split
warnings.simplefilter(action="ignore",category=FutureWarning)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
plt.figure(figsize=(4,3))
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

mse=sklearn.metrics.mean_squared_error(Y_test,Y_pred)
print(mse)

plt.figure(figsize=(4, 3))
plt.hist(boston.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()

for index, feature_name in enumerate(boston.feature_names):
  plt.figure(figsize=(4, 3))
  plt.scatter(boston.data[:, index], boston.target)
  plt.ylabel('Price', size=15)
  plt.xlabel(feature_name, size=15)
  plt.tight_layout()
 
from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor()
clf.fit(X_train,Y_train)

predicted = clf.predict(X_test)
expected = Y_test

plt.figure(figsize=(4, 3))
plt.scatter(predicted, expected)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()

mse=sklearn.metrics.mean_squared_error(predicted,expected)
print(mse)
