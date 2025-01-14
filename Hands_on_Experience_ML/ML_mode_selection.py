# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


# import models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# load data
heart_data = pd.read_csv('/home/muhammed-shafeeh/AI_ML/Ai-and-Ml/ML_heart_deseas_prediction/heart_disease_data.csv')

# print first 5 rows
heart_data.head()

# shape of the data
heart_data.shape

# any missing values
heart_data.isnull().sum()

# statistics
heart_data.describe()

# check the distribution of target variable
heart_data['target'].value_counts()

# 0 healthy heart
# 1 diseased heart
# split the data into features and target
X = heart_data.drop('target', axis=1)
Y = heart_data['target']

print(X)
print(Y)

# split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)
print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

print(X.shape, X_train.shape, X_test.shape)
print(Y_test.shape)

# Comparing the models with defualt hyperparameters values using cross validation

#**1. Comparing the models with default hyperparameter values  using cross validation

models = [
    LogisticRegression(max_iter=10000),
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier(random_state=0)]


def compare_models():
    for model in models:
        
        cv_score = cross_val_score(model, X_train, Y_train, cv=5)
        mean_score = cv_score.mean()*100
        mean_score = round(mean_score, 2)
        
        print("cross_val_score: " , model, " = ", cv_score)
        print("accuracy_score of the ", model, " = ", mean_score )
        print("\n")
        
        
compare_models()


#Inferance for the heart diseas dataset, #**Random Forest Classifier has the Highest accuracy value with default hyperparameter

