<H3>ENTER YOUR NAME</H3>
Sana Fathima H
<H3>ENTER YOUR REGISTER NO.</H3>
212223240145
<H3>EX. NO.1</H3>
<H3>DATE</H3>
26.08.2025
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle.

## EQUIPMENTS REQUIRED:
Hardware – PCs

Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:

STEP 1:Importing the libraries.<BR>

STEP 2:Importing the dataset.<BR>

STEP 3:Taking care of missing data.<BR>

STEP 4:Encoding categorical data.<BR>

STEP 5:Normalizing the data.<BR>

STEP 6:Splitting the data into test and train.<BR>

STEP 7:Print the values.

STEP 8:End the program.

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv")
df
df.isnull().sum()
df.duplicated()
print(df['CreditScore'].describe())
df.info()
df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df
Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1
x=df1.iloc[:,:-1].values
print(x)
y=df1.iloc[:,-1].values
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
### Dataset
<img width="1281" height="582" alt="Screenshot 2025-08-26 084132" src="https://github.com/user-attachments/assets/2b2426e7-865e-4e1c-8a1c-c549579b96d9" />

### Null Count
<img width="1272" height="431" alt="Screenshot 2025-08-26 084157" src="https://github.com/user-attachments/assets/ed191f9e-d957-4eaf-9fec-3dffca08f947" />

### Duplicated Values
<img width="1268" height="347" alt="Screenshot 2025-08-26 084224" src="https://github.com/user-attachments/assets/85280246-144f-4d33-b694-278d1587c04c" />

### Feature.describe()
<img width="1276" height="266" alt="Screenshot 2025-08-26 084243" src="https://github.com/user-attachments/assets/21326235-77d4-4dca-af68-3ec4e38d5fab" />

### df.info()
<img width="1271" height="591" alt="Screenshot 2025-08-26 084307" src="https://github.com/user-attachments/assets/a1597c10-265c-4737-9c79-f473e1367b16" />

### df.drop()
<img width="1269" height="580" alt="Screenshot 2025-08-26 084343" src="https://github.com/user-attachments/assets/762f439d-e051-4da0-97cc-be223e4d6180" />

### Feature Transformation
<img width="1272" height="576" alt="Screenshot 2025-08-26 084404" src="https://github.com/user-attachments/assets/0fb57237-3108-4d34-82c8-3cdbe5d9adf9" />

### X Values
<img width="1269" height="377" alt="Screenshot 2025-08-26 084434" src="https://github.com/user-attachments/assets/adbaf264-60c2-4ca1-a955-9cab6bf2719d" />

### Y Values
<img width="713" height="63" alt="Screenshot 2025-08-26 084457" src="https://github.com/user-attachments/assets/966e9749-a5f9-484a-83a5-869e1f1c3d74" />

### Split the Dataset into train and test data
#### x_train Values
<img width="1085" height="219" alt="Screenshot 2025-08-26 084517" src="https://github.com/user-attachments/assets/dbd908f0-a4af-41a1-8803-7c97b73992b2" />
<img width="876" height="103" alt="Screenshot 2025-08-26 084534" src="https://github.com/user-attachments/assets/feafed96-a541-4e58-be90-db282fa9ddd9" />

#### x_test Values
<img width="1160" height="225" alt="Screenshot 2025-08-26 084552" src="https://github.com/user-attachments/assets/d92cd74c-fca1-42e1-9822-7632a3c23481" />
<img width="810" height="100" alt="Screenshot 2025-08-26 084709" src="https://github.com/user-attachments/assets/ae569bbd-3e81-43b6-9f23-9495b856c153" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


