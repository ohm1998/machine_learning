'''
Electricity Problem on HackerEarth
'''


import numpy as np
import pandas as pd
from sklearn import linear_model


m = int(input())
inp = []
for i in range(m):
	inp.append(list(map(int,input().split(' '))))

inp = np.array(inp)

X = np.column_stack((inp[:,0],inp[:,1],inp[:,2]))
Y = inp[:,3]

reg = linear_model.LinearRegression()

reg.fit(X,Y)

inp_pred = list(map(int,input().split(' ')))

nancy_pri = float(input())

inp_pred = np.column_stack((inp_pred[0],inp_pred[1],inp_pred[2]))


pred = reg.predict(inp_pred)

if((pred - nancy_pri) >= 2000):
	flag=1
else:
	flag=0

print(pred,' ',flag)
