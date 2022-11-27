import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

sc = pickle.load(open('Scaled.pkl', 'rb'))
rfr = pickle.load(open('ML_model.pkl', 'rb'))

iw = int(input('Enter IW: '))
_if = int(input('Enter IF: '))
vw  = float(input('Enter VW: '))
fp  = int(input('Enter FP: '))

X_input = []
for i in range(4):
    x_input = []
    x_input.append(iw)
    x_input.append(_if)
    x_input.append(vw)
    x_input.append(fp)
    x_input.append(i)
    X_input.append(x_input)

X_input = np.array(X_input)
X_input = sc.transform(X_input)
output = rfr.predict(X_input)

i = 0
for depth, width in output:
    i += 1
    print(i, ' section:')
    print('Depth = ', depth, '; Width = ', width)