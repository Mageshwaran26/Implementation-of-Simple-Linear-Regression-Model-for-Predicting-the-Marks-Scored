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
Developed by: Mageshwaran T.A
RegisterNumber: 212224230146 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,reg.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```
## Output:
Head values

![image](https://github.com/user-attachments/assets/f5421985-02a0-4fe7-8ca2-cbac0280966d)


Tail values

![Screenshot 2025-03-15 102600](https://github.com/user-attachments/assets/6c2fb45d-a36e-4346-b356-ba8e30e42086)

Array value of X

![image](https://github.com/user-attachments/assets/6dd27667-9cbd-4796-b308-e7d04cf57745)

Array values of Y

![image](https://github.com/user-attachments/assets/0ff857f7-707d-4b0c-b464-4462a17e0beb)

Predicted Y values are

![image](https://github.com/user-attachments/assets/190862ee-c3e5-4854-8548-02c13dfd41ed)

Actual Y values are

![image](https://github.com/user-attachments/assets/e8a4d887-ed2f-41c7-a9e8-f7c717bc22be)

Training Set Graph

![image](https://github.com/user-attachments/assets/44fe5899-12f7-4878-b1ed-0113d933f069)

Testing Set Graph

![image](https://github.com/user-attachments/assets/3a94b6cb-bc22-44f5-be48-a82997b2559a)

Values of MSE,MAE,RMSE

![image](https://github.com/user-attachments/assets/a2312ccd-32c8-404d-9813-2742a51aa01c)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
