<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

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
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))




## OUTPUT:

Dataset:
<img width="1562" height="520" alt="image" src="https://github.com/user-attachments/assets/71b026a5-1b8d-496d-93f4-23c1395d9819" />
X Values:
<img width="721" height="251" alt="image" src="https://github.com/user-attachments/assets/4c43e203-6c41-43e5-a3e0-07a9f65292d4" />
Y Values:
<img width="325" height="119" alt="image" src="https://github.com/user-attachments/assets/220a52dd-4ea4-42ff-92e0-1df81370f707" />
Null Values:
<img width="230" height="538" alt="image" src="https://github.com/user-attachments/assets/511e1aa3-4b81-45d4-bfd7-962b6425d656" />
Duplicated Values:
<img width="295" height="710" alt="image" src="https://github.com/user-attachments/assets/9df9555a-9a7c-4891-8773-c82973ac7573" />
Description:
<img width="1037" height="237" alt="image" src="https://github.com/user-attachments/assets/2607d677-897b-4fe8-992f-2913b05c60b4" />
Normalized Dataset:
<img width="916" height="746" alt="image" src="https://github.com/user-attachments/assets/13d1137a-c874-405f-8f35-d384e2b618f4" />
Training Data:
<img width="873" height="235" alt="image" src="https://github.com/user-attachments/assets/52cee96b-ac1f-41a1-b4f9-df4bb4724a5b" />
Testing Data:
<img width="862" height="212" alt="image" src="https://github.com/user-attachments/assets/6921d9ec-bcc3-4c19-a22e-758d9cf15534" />

<img width="871" height="82" alt="image" src="https://github.com/user-attachments/assets/020e948f-44eb-4707-82a9-9c48b88b9e0c" />









## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


