# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

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
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/38da07ca-4480-4cd2-91d2-3a10e58d967e)
```
df.head()
```
![image](https://github.com/user-attachments/assets/02811e67-190b-45e6-b8e6-e5b747da8e5b)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/ef5669aa-d2e1-440a-a62b-f0f8f81a76bd)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/db5c9b43-576d-4924-af4f-c1ac3067d5ab)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/8291e4f2-eba2-46e4-91d7-6cc9b3b3e198)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/31c75e25-2852-4b4d-a366-84524213bdb6)
```
from sklearn.preprocessing import Normalizer
scale=Normalizer()
df[['Height','Weight']]=scale.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/1eed9eef-cc13-4b83-8b3c-be3677080db4)
```
from sklearn.preprocessing import MaxAbsScaler
scalen=MaxAbsScaler()
df[['Height','Weight']]=scalen.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/bed6623c-a343-4d63-a593-beef886095a0)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/7ade8c05-3083-428f-afc0-d2b4c7b39b8e)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv')
data
```
![image](https://github.com/user-attachments/assets/401cb4c1-f665-4e21-bee3-06c78af7a5bf)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/beed9626-c888-4516-be6d-96b75f17d58b)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/a8d7f9d5-d3f8-4eef-b945-05db4cf7914b)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/3dd9a570-889d-4dd2-ba25-de2a443f10f6)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/16825fcb-f893-4a6c-b66f-5a1702d3b164)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/80b2bd45-5998-44a0-a847-55c994c9ddf6)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/130fdcaf-9eda-438c-9fce-ce64aa2122e4)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/17ab51e6-6093-4a7b-970d-fa3e3c776f0f)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/cea9befd-9e6c-4a15-995e-ed3b089e8c73)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/ac20350f-5858-40ce-99b7-83c14becedcd)
```
x=new_data[features].values
x
```
![image](https://github.com/user-attachments/assets/62fcf631-974e-4245-8770-4d42b408bbb1)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/2cf736f8-09ed-4c16-869c-47f0fe437cbd)
```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/23a47c1e-c77f-496f-b660-39e08c42766f)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/0747530e-ed73-4047-ad6e-d49e8607abc2)
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/user-attachments/assets/eec1b332-78dc-4d16-acf2-885c2d8af0b4)
```
data.shape
```
![image](https://github.com/user-attachments/assets/06fcf027-37e0-4108-bd92-5f2ca4833ae1)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/32fb7d52-24f9-4f58-a561-3a6ba2aefc1c)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/656ff68d-d3f2-4f6b-aeb6-92b9cc23eb19)
```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/user-attachments/assets/c0d24942-dca5-45b6-a425-06605b50c6af)
```
chi2,p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/67c43f52-cb13-489c-8008-1701e2946d77)

# RESULT:
Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
