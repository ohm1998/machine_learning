import numpy as np
import pandas as pd
from sklearn import linear_model

file_name = "197_cpu_act.tsv"

df1 = pd.read_csv(file_name,sep="\t")



dfshape = df1.shape


len_x = (dfshape[1]-1)

print(len_x)

X = np.array(df1.iloc[:,0])


	
for i in range(1,len_x):
	X = np.column_stack((X,df1.iloc[:,i]))

Y = np.array(df1.iloc[:,len_x])

Y = np.reshape(Y,(-1,1))

reg = linear_model.LinearRegression()

reg.fit(X,Y)

print("Coefficeint(s): ",reg.coef_,"  Intercept: "reg.intercept_)
