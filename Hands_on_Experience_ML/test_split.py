# Train Test Split
# data preprocessing
# data ----> data_preprocessing ----> data analysis ----> model building ----> train test split ----> machine learning model ----> model evaluation

# data set
# testing and training data set

# need of evaluation to test the model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetis_data = pd.read_csv(
    "/home/muhammed-shafeeh/ML_AI/Ai-and-Ml/data/ai_and_ml/diabetes.csv"
)
diabetis_data.head()

diabetis_data["Outcome"].value_counts()

# 0 - Non Diabetic
# 1 - Diabetic

diabetis_data.groupby("Outcome").mean()

# splitting the data and labels
X = diabetis_data.drop(columns="Outcome", axis=1)
Y = diabetis_data["Outcome"]
X.head()
Y.head()

# standardizing the data
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
print(standardized_data)

# splitting the data into training and testing data
X_test, X_train, Y_test, Y_train = train_test_split(
    standardized_data, Y, test_size=0.2, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)
