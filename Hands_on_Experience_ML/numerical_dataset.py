# application: all darta processing and manipulation functions using a case study


# importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# data collection and processing

# loading the dataset
diabetes_data = pd.read_csv("data/ai_and_ml/diabetes.csv")
diabetes_data.head()

# number of rows and columns
diabetes_data.shape

# count the outcome of the target variable
diabetes_data["Outcome"].value_counts()

# 0 -> Non-Diabetic
# 1 -> Diabetic

# separating the data and label
X = diabetes_data.drop(columns="Outcome", axis=1)
Y = diabetes_data["Outcome"]
print(X)
print(Y)

# data standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)
print(standardized_data)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    standardized_data, Y, test_size=0.2, random_state=2
)
print(X.shape, X_train.shape, X_test.shape)


"""
# equal number of samples from both classes
diabetic = diabetes_data[diabetes_data["Outcome"] == 1]
non_diabetic = diabetes_data[diabetes_data["Outcome"] == 0]

diabetic.shape, non_diabetic.shape

# taking equal number of samples from both classes
non_diabetic_sample = non_diabetic.sample(n=268)
"""
