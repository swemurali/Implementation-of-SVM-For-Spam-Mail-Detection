# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M.Suwetha
RegisterNumber:  212221230112
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![v1](https://user-images.githubusercontent.com/94165336/203983548-0ff6c8d2-6fac-47e5-a671-8d85ed27e2d7.png)

![v2](https://user-images.githubusercontent.com/94165336/203983564-6ce6f18c-e6b4-467e-9733-470120b9133b.png)

![v3](https://user-images.githubusercontent.com/94165336/203983582-59a502a9-cd20-4cfe-8f87-534cd0bcee26.png)

![v4](https://user-images.githubusercontent.com/94165336/203983602-3a20c1fa-2e95-460c-8cb2-2f5f5462c631.png)

![v5](https://user-images.githubusercontent.com/94165336/203983615-d845aca9-43e6-4438-8bc4-536cebea4877.png)

![v6](https://user-images.githubusercontent.com/94165336/203983632-502e9b15-3818-414c-af12-f7f2ace3bb16.png)

![v7](https://user-images.githubusercontent.com/94165336/203983651-cf26b745-8f70-416d-a8a1-729e9d6e2e32.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
