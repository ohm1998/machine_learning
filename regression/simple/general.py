import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math
excel_name = "slr12.xls"

def ypoints(x,slope,intercept):
	y = ((x*slope)+intercept)
	return y

df1 = pd.read_excel(excel_name)

break_point = math.floor((0.8*(df1.shape[0])))
end_point = df1.shape[0]
print(break_point,end_point)
x_train = df1.iloc[0:break_point,0].values
y_train = df1.iloc[0:break_point,1].values
x_test = df1.iloc[break_point:end_point,0].values
y_test = df1.iloc[break_point:end_point,1].values

reg = linear_model.LinearRegression()

x_train = np.reshape(x_train,(-1,1))
y_train = np.reshape(y_train,(-1,1))
x_test  = np.reshape(x_test,(-1,1))
y_test  = np.reshape(y_test,(-1,1))

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

print(reg.score(x_test,y_test))

plt.scatter(x_train,y_train,color="red",label="Training Set")
#plt.scatter(x_test,y_pred,color="blue")
plt.scatter(x_test,y_test,color="green",label="Testing Set")

if(min(x_train)>min(x_test)):
	minimum_x = min(x_test)
else:
	minimum_x = min(x_train) 

if(max(x_train)>max(x_test)):
	maximum_x = max(x_train)
else:
	maximum_x = max(x_test)

x = np.linspace(minimum_x-x_train[0],maximum_x+x_train[0],9000)

x = np.reshape(x,(-1,1))

y = ypoints(x,reg.coef_,reg.intercept_)


plt.plot(x,y,color="black")
plt.legend()
plt.show()