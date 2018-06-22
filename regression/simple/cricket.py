import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def ypoints(x,slope,intercept):
	y = ((x*slope)+intercept)
	return y

df1 = pd.read_excel("cricket.xls")



x_train = df1.iloc[0:10,0].values
y_train = df1.iloc[0:10,1].values
x_test = df1.iloc[10:16,0].values
y_test = df1.iloc[10:16,1].values


reg = linear_model.LinearRegression()

x_train = np.reshape(x_train,(-1,1))
y_train = np.reshape(y_train,(-1,1))
x_test  = np.reshape(x_test,(-1,1))
y_test  = np.reshape(y_test,(-1,1))

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

print(reg.score(x_test,y_test))

plt.scatter(x_train,y_train,color="red")
plt.scatter(x_test,y_pred,color="blue")
plt.scatter(x_test,y_test,color="green")
x = np.linspace(11,21,2000)

x = np.reshape(x,(-1,1))

y = ypoints(x,reg.coef_,reg.intercept_)

print(x,y)

plt.plot(x,y,color="black")
plt.show()