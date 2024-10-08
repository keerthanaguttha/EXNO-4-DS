# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
```
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.
```
# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas  as  pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=["?"])
data
```

![image](https://github.com/user-attachments/assets/24981339-0501-4a80-b21b-ae7f4051304c)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/ae5762a3-9eb1-4d6b-be19-32b8948b6984)

```
missing= data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/a87db41c-5c7e-48ce-800c-19f1d4d1b7fb)

```
data2=data.dropna(axis=0)
data2
```

![image](https://github.com/user-attachments/assets/7eeada54-2e56-4c31-b087-6e9ba4034eeb)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/user-attachments/assets/e7994785-c72b-4655-b240-353df381820b)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/761ff312-510f-487b-9455-5e22f8be17b8)

```
data2
```

![image](https://github.com/user-attachments/assets/19fa0e2c-16cb-4b3b-9f6e-779ed0cb882b)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/b7470cbd-6627-4950-bf3e-d664c1eb2304)

```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/user-attachments/assets/c245246c-2bae-43ef-8ee5-b5a242f6a0c3)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/user-attachments/assets/5ae1ccc1-28fa-4791-bf72-beeec574ca15)

```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/dc583e86-82e6-4820-9586-3b3b7e386ecb)

```
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/51287de5-68f1-42f3-8232-db04ad2ffcb0)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```

![Screenshot 2024-10-08 103101](https://github.com/user-attachments/assets/2473533a-1c53-4cf3-95c5-b7891fb818c3)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y,prediction)
print(confusionMatrix)
```

![image](https://github.com/user-attachments/assets/f069202d-84d1-4c88-86c2-50d2aa88b665)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/c7011e1f-2703-416a-9965-5dd64d6bfd33)

```
print('Misclassified samples: %d' %  (test_y != prediction).sum())
```

![image](https://github.com/user-attachments/assets/42124bec-18c6-4315-990d-bcebf5585b9f)

```
data.shape
```

![image](https://github.com/user-attachments/assets/ac0949a6-575a-41b5-94ad-c57e29c0f1dd)

# RESULT:
Thus, the program for feature selection and scaling has been implemented.
