# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries which are used for the program.
2. Load the dataset.
3. Check for null data values and duplicate data values in the dataframe.
4. Apply logistic regression and predict the y output.
5. Calculate the confusion,accuracy and classification of the dataset.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HEMASONICA.P
RegisterNumber: 212222230048

import pandas as pd
df=pd.read_csv("Placement_Data.csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![image](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/8d61da1d-6b43-474b-90d3-0a2c4ce4e895)

![Screenshot 2023-08-31 093631](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/6eda955e-54d3-46f9-aea7-b9f756aad8e4)

![Screenshot 2023-08-31 093640](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/e4e1589c-3892-4cbf-b56e-b0e1a87957f9)

![Screenshot 2023-08-31 093647](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/79e50669-816f-4168-9601-c5e4fc1fe3a3)

![Screenshot 2023-08-31 093703](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/3fa84ca6-67f9-493b-b7af-0671840b1840)

![Screenshot 2023-08-31 093717](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/e78928c4-226b-44be-b124-d83b5ec653cc)

![Screenshot 2023-08-31 093730](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/5452b5a1-82fe-4956-8107-bd374bb9cfd7)

![Screenshot 2023-08-31 093744](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/8f52299e-9254-4e04-a145-54220d84a0a8)

![Screenshot 2023-08-31 093759](https://github.com/Hemasonica774/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118361409/e60f735e-5654-45e4-b22c-aabe93de45b6)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
