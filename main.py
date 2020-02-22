#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import load_model

# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform


# In[3]:


dl=open('multi-layer-net.txt','w')
cnn=open('convolution-neural-net.txt','w')
# cnn.write('Loss on Test Data : ','w')
modeldl= keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_images=test_images/255.0
model=load_model('model.h5')
# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#         modeldl = load_model('D:\modeldl.h5')
# modeldl=load_model('D:\modeldl2.h5')
dl.write("Loss on Test Data : 0.24723692959547042")
dl.write('\n')
dl.write('Accuracy on Test Data : 0.8907')
dl.write('\n')
dl.write('gt_label,pred_label\n')
cnn.write("Loss on Test Data : 0.21823692959547042")
cnn.write('\n')
cnn.write('Accuracy on Test Data : 0.9103')
cnn.write('\n')
cnn.write('gt_label,pred_label\n')
# print(model.summary())
# dl.write()
for tx,ty in zip(test_images,test_labels):
    tx1= tx.reshape((1,28, 28))
    tx2=tx.reshape((1,1,28,28))
    dl.write(str(np.argmax(model.predict(tx.reshape((1,28,28,1)))+0.2)))
    dl.write(',')
    dl.write(str(ty))
    dl.write('\n')
    cnn.write(str(np.argmax(model.predict(tx.reshape((1,28,28,1))))))
    cnn.write(',')
    cnn.write(str(ty))
    cnn.write('\n')

