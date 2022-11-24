from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

path = input('Enter the path to the file with data on the size of the weld: ')
data = pd.read_csv(path)

section = []
for i in range(18):
  section.append(0)
  section.append(1)
  section.append(2)
  section.append(3)
data['section'] = section

X = np.array(data.drop(['Depth', 'Width'], axis=1))
y = np.array(data.drop(['IW', 'IF', 'VW', 'FP', 'section'], axis=1))
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=56)

rfr = RandomForestRegressor(100, criterion='squared_error', max_depth=5, min_samples_split=3, min_samples_leaf=1)
rfr.fit(X_train, y_train)
y_predict = rfr.predict(X_test)

result = pd.DataFrame()
y_predict_depth = []
y_predict_width = []
for i in y_predict:
    y_predict_depth.append(i[0])
    y_predict_width.append(i[1])
y_true_depth = []
y_true_width = []
for i in y_test:
    y_true_depth.append(i[0])
    y_true_width.append(i[1])
result['y_predict_depth'] = y_predict_depth
result['y_predict_width'] = y_predict_width
result['y_true_depth'] = y_true_depth
result['y_true_width'] = y_true_width
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print(result)

print('\nMean absolute error = ', mean_absolute_error(y_test, y_predict))
print('Mean absolute percentage error = ', mean_absolute_percentage_error(y_test, y_predict))
print('Mean squared error = ', mean_squared_error(y_test, y_predict))
