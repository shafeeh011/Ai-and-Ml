# import basic libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# download the dataset
'''
from kaggle.api.kaggle_api_extended import KaggleApi
KaggleApi().authenticate()
KaggleApi().dataset_download_files('barelydedicated/bank-customer-churn-modeling', unzip=True)
'''

# read the data
df = pd.read_csv('/home/muhammed-shafeeh/AI_ML/Ai-and-Ml/Deep_Learning_YT/Customer_churn_prediction/bank_customer_churn.csv')
df.head()

# devide the data to indipendent and dependent features
X = df.iloc[:,3:13]
Y = df.iloc[:,-1]

X.head()
Y.head()

# feature engineering
# get the dataset unique values of each column

X.select_dtypes(include='object').columns

# create a fuction for this which are the categorical
def categorical_columns(df):
    categorical_col = df.select_dtypes(include='object').columns
    for column in categorical_col:
        print(f'{column} : {df[column].unique()}')
        
categorical_columns(X)

