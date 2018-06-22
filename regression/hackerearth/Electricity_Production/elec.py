import numpy as np
import pandas as pd
from sklearn import linear_model
import csv

df1 = pd.read_csv("train_data.csv")

print(df1.head())

X = np.column_stack((df1.iloc[:,0].values,df1.iloc[:,1].values,df1.iloc[:,2].values,df1.iloc[:,3].values))
print(X[0:10])
Y = np.column_stack((df1.iloc[:,4].values))

Y = np.reshape(Y,(-1,1))


lr = linear_model.LinearRegression()

lr.fit(X,Y)

df2 = pd.read_csv("test_data.csv")

X = np.column_stack((df2.iloc[:,0].values,df2.iloc[:,1].values,df2.iloc[:,2].values,df2.iloc[:,3].values))


prediction = lr.predict(X)

myfile  = open('submit.csv','w')

with myfile:
	writer = csv.writer(myfile)
	writer.writerows(prediction)
