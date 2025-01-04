# data in the form of textual data
# converting the textual data into numerical data
# Bag of Words (BoW): list of unique wordd in the text corpq
# Term Frequency-Inverse Document Frequency (TF-IDF): To count the frequency(the number of words) of words in the text
# Term Frequency (TF): The number of times a word appears in a document divided by the total number of words in the document
# Inverse Document Frequency (IDF): The log of the number of documents divided by the number of documents that contain the word W
# the IDF valeu increases as the number of documents that contain the word decreases
# TF-IDF value of a term = TF * IDF

"""
about the dataset:
1.Id: unique id for a news article
2.Tiile: title of the news article
3.Author: author of the news article
4.Text: text of the news article
5.Label: 1-> fake news, 0-> real news
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

# importing the dataset
news_dataset = pd.read_csv(
    "/home/muhammed-shafeeh/ML_AI/Ai-and-Ml/data/ai_and_ml/train.csv"
)
news_dataset.head()
news_dataset.shape
news_dataset["label"].value_counts()


# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the missing values with empty strings
news_dataset = news_dataset.fillna("")

# merging the author name and news title
news_dataset["content"] = news_dataset["author"] + " " + news_dataset["title"]
news_dataset.head()

print(news_dataset["content"])

# separating the data and label
X = news_dataset.drop(columns="label", axis=1)
Y = news_dataset["label"]
print(X)
print(Y)
print(X.shape)
print(Y.shape)


# TF-IDF (Term Frequency-Inverse Document Frequency)
# converting the textual data to feature vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X)
