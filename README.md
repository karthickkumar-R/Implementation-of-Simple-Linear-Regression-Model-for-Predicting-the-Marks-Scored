# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICKKUMAR R
RegisterNumber: 212223040087 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
print(df.head())
df.tail()
print(df.tail())
X=df.iloc[:,:-1].values
X
print(X)
Y=df.iloc[:,1].values
Y
print(Y)

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
print(Y_pred)
Y_test
print(Y_test)

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:

![1](https://github.com/user-attachments/assets/eaa78e4e-6e48-4105-9f02-c18f90b52dbd)

![2](https://github.com/user-attachments/assets/2d8b1832-455f-40fa-a6ff-83482a1a9a59)

![3](https://github.com/user-attachments/assets/d99ce812-d0d0-42c1-821e-1922763b67cb)

![4](https://github.com/user-attachments/assets/f0e37d45-4280-401e-8d22-49b1d38b3636)

![5](https://github.com/user-attachments/assets/48795f4d-ef81-4e5f-ad4f-ed203ad89b72)

![6](https://github.com/user-attachments/assets/dfdb545c-70c0-425e-8b3e-f9ce3fbd02b0)

![7](https://github.com/user-attachments/assets/cb15fd94-7282-472e-995f-39939290fc24)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
