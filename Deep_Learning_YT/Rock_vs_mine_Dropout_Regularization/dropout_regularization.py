#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# read the data
df = pd.read_csv('/home/muhammed-shafeeh/AI_ML/Ai-and-Ml/Deep_Learning_YT/Rock_vs_mine_Dropout_Regularization/sonar_data.csv', header=None)

# first 5 rows of the data
df.head()

df.columns

df[60].value_counts()

# 1 - M
# 0 - R
# label encoding using sklearn
label_encoders = LabelEncoder()
label = label_encoders.fit_transform(df[60])
df[60] = label
df.head()


X = df.drop(columns=60, axis=1)
Y = df[60]

Y.head()

df[60].value_counts()



