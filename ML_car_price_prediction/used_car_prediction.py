
# **Used Car Price Prediction**
# 
# using two method 
# 
# 1. Linear Regression
# 2. LASSO Regression 

# %%


# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# %%
# read the data
car_data = pd.read_csv('home/muhammed-shafeeh/ML_AI/AI and ML/ML_car_price_prediction/car data.csv')

# %%
# first 5 rows of the data
car_data.head()

# %%
# checking the number of rows and columns
car_data.shape

# %%
# checking for missing values
car_data.isnull().sum()

# %%
# information about the data
car_data.info()

# %%
# checking total numer of catogorical data in a fuel type, transmission and seller type
car_data['Fuel_Type'].value_counts()


# %%
car_data['Transmission'].value_counts()

# %%
car_data['Seller_Type'].value_counts()

# %%
car_data['Owner'].value_counts()

# %%
# encoding the catogorical data to numerics
car_data['Fuel_Type'] = car_data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})




# %%
car_data['Transmission'] = car_data['Transmission'].map({'Manual':0, 'Automatic':1})


# %%

car_data['Seller_Type'] = car_data['Seller_Type'].map({'Dealer':0, 'Individual':1})

# %%

car_data.head(20)

# %%
car_data.info()

# %%
# spliting the data into traing and test data
X = car_data.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_data['Selling_Price']

# %%
X.head(50)

# %%
# print shape of X and Y
print(X)
print(Y)

# %%
# split in to train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# %%
# print shape of X_train and Y_train

print(X_train.shape)
print(Y_train.shape)

# %%
# model linear regression
linear_model = LinearRegression()


# %%
linear_model.fit(X_train, Y_train)

# %%
# prediction on training data
Y_train_predict = linear_model.predict(X_train) 

# %%
# R squared value
error_score = metrics.r2_score(Y_train, Y_train_predict)
print("error score= ", error_score)

# %%
# visualize the actual and predicted values
plt.scatter(Y_train, Y_train_predict)
plt.xlabel("actual values")
plt.ylabel("predicted values")
plt.title("actual values vs predicted values")

# %%



# %%



