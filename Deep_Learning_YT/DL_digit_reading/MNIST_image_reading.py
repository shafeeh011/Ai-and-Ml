
# **MNIST imafe reading**

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
import cv2 

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix
from xgboost import train


# loading the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# visualizing the data
plt.imshow(train_images[0], cmap='gray')
plt.show()

# normalizing the data
train_images = train_images / 255
test_images = test_images / 255

plt.imshow(train_images[0], cmap='gray')
plt.show()
print(train_labels[0])

print(train_images[0])
print(train_images[0].shape)

# image labels
print(train_labels[0])
print(train_labels.shape)

# labels unique values
print(np.unique(train_labels))
print(np.unique(test_labels))


# ** we can use image have the same dimension or oneHotEncoding **

# here use image have the same dimension
# building the nueural network
# flatten the images

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')  # softmax for classification for multiclasses
    ])


# compling the model
model.compile(
    optimizer= 'adam',
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# training the model
model.fit(train_images, train_labels, epochs=10)  

# testing or evaluating the model
model.evaluate(test_images, test_labels)  

# predicting the model
    
# plot the first the image of X_test
plt.matshow(test_images[18])
plt.show()

Y_pred = model.predict(test_images)


print(Y_pred[2])
label_for_the_first_image = np.argmax(Y_pred[2])

print(test_labels[2])

# create a for loop for all the images in Y_pred

Y_pred_label = [np.argmax(i) for i in Y_pred]

print(Y_pred_label[:5])

print(test_labels[:5])

# confusion matrix
confusion_matrix = confusion_matrix(test_labels, Y_pred_label)

print(confusion_matrix)

# visualize the confusion matrix
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

'''
# predictive system

input_image_path = input('Enter the image path: ')
#print(input_image_path)

img= cv2.imread(input_image_path)
#plt.imshow(img)
img.shape
# resixe the image
img = cv2.resize(img, (28,28))
img.shape
# conver to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img.shape

# standardize the image
img = img / 255
img.shape

img = img.reshape(1,28,28)
img.shape

# predct the image
prediction = model.predict(img)
print(prediction)
label_for_the_first_image = np.argmax(prediction)
print(label_for_the_first_image)
'''

# user imput and prediction
import os

input_image_path = input('Enter the image path: ')

if not os.path.exists(input_image_path):
    print("Error: File not found. Please check the file path.")
else:
    img= cv2.imread(input_image_path)
    plt.imshow(img)
    img = cv2.resize(img, (28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = img.reshape(1,28,28)
    prediction = model.predict(img)
    label_for_the_first_image = np.argmax(prediction)   
    print(label_for_the_first_image)




