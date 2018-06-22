import numpy as np
import pandas as pd
from sklearn import linear_model

df1 = pd.read_csv("192_vineyard.tsv",sep="\t")

print(df1.head())

X = np.column_stack((df1.iloc[0:45,0].values,df1.iloc[0:45,1].values))

print(df1.shape)

Y = np.reshape(df1.iloc[0:45,2].values,(-1,1))

x = np.column_stack((df1.iloc[45:52,0].values,df1.iloc[45:52,1].values))
y = np.reshape(df1.iloc[45:52,2].values,(-1,1))


lr = linear_model.LinearRegression()

lr.fit(X,Y)

print(lr.coef_)
print(lr.intercept_)

print(lr.score(x,y))