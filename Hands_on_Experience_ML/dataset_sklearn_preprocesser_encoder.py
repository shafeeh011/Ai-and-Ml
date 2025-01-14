import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# first 5 rows of the dataset
cancer_data = pd.read_csv("/home/muhammed-shafeeh/AI_ML/Ai-and-Ml/data/ai_and_ml/cancer_data.csv")
cancer_data.head()

# finding the count of different labels
cancer_data["diagnosis"].value_counts()

# load the encoder function
label_encoder = LabelEncoder()

labels = label_encoder.fit_transform(cancer_data.diagnosis)

# appendoing the labels to the dataset
cancer_data["target"] = labels

cancer_data.head()

# 1 - Malignant
# 0 - Benign

cancer_data["target"].value_counts()

# iris dataset
iris_data = pd.read_csv(
    "/home/muhammed-shafeeh/ML_AI/Ai-and-Ml/data/ai_and_ml/iris_data.csv"
)
iris_data.head()
iris_data.tail()

# species count
iris_data["Species"].value_counts()

# liading the encoder function
label_encoder_1 = LabelEncoder()

iris_labels = label_encoder_1.fit_transform(iris_data.Species)

iris_data["target"] = iris_labels
iris_data.head()

iris_data["target"].value_counts()

# 0 - setosa
# 1 - versicolor
# 2 - virginica
