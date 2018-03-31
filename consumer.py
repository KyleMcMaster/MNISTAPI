# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:42:12 2018

@author: kyler
"""

import keras
from keras.datasets import mnist
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#get sample for API to classify and valid label
sample = x_test[0:1]
label = y_test[0:1]

#let's visualize the sample before sending to API.
#visualization of input sample (https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot)
# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = sample.reshape((28, 28))

#Plot
plt.title('Sample sent to API')
plt.imshow(pixels, cmap='gray')
plt.show()

#API call
#set request headers
header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

#post request
resp = requests.post("http://127.0.0.1:5000/predict", data = json.dumps(sample.tolist()), headers= header)

#breakdown response
data  = json.loads(resp.json()['prediction'])
prediction = data[0]

#print predicition vs results
print('Prediction: ', np.argmax(prediction))
print('Actual: ', np.argmax(label))
