# handling imbalance in the dataset

# Importing necessary libraries
import numpy as np
import pandas as pd

# importing the dataset
credit_card_data = pd.read_csv("/home/muhammed-shafeeh/ML_AI/Ai-and-Ml/data/ai_and_ml/credit_data.csv")
credit_card_data.head()


# distribution of the the classes
credit_card_data["Class"].value_counts()
# this is a highly imbalanced dataset
# 0 - Non Fraud
# 1 - Fraud

# separating non fraud and fraud data 

non_fraud = credit_card_data[credit_card_data["Class"] == 0]
fraud = credit_card_data[credit_card_data["Class"] == 1]   

print(non_fraud.shape)
print(fraud.shape)

# Under Sampling
# taking equal number of samples from both classes (takig the same number of samples from both classes) 
# the numbet od fraud samples is 492
# so we take 492 non fraud samples

non_fraud_sample = non_fraud.sample(n = 492)
print(non_fraud_sample.shape)

# combining the samples
new_dataset = pd.concat([non_fraud_sample, fraud], axis = 0)
new_dataset.head()
new_dataset.tail()


new_dataset["Class"].value_counts()
import kaggle 