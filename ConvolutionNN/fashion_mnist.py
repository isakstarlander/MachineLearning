# Train a CNN on the MNIST fashion data. The data consists of 10 classes of fashion images such as
# shorts, dresses, shoes, purses, etc. The images replace the handwritten digits in the classic
# MNIST dataset. This change makes it harder to get a high score and more closely reflects real world
# usage of image classification. At the same time it is small enough for the average PC to train
# in a short time. https://github.com/zalandoresearch/fashion-mnist has more info on Fashion MNIST
from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Suppress warning and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Number of classes
num_classes = 10

# batch size and # of epochs
batch_size = 128
epochs = 24

# input image dimensions
img_rows, img_cols = 28,28

# the data shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Deal with format issues between different backends. Some put # of channels in the image before 
# the width and height
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Type convert and scale the test and training data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices. I.e. one-hot encoding
# 3 -> 0 0 0 1 0 0 0 0 0 0 and 1 -> 0 1 0 0 0 0 0 0 0 0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))                       # A posted paper shows that removing early pooling layers improve performance
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))    # using 64 instead of 32 has shown better results
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Define compile to minimize the categorical loss, use ada delta optimizer and
# optimize with regards to accuracy
model.compile(loss=keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adadelta(), 
    metrics=['accuracy'])

# Train the model and test/validate the mode with the test data after each cycle (epoch)
# through the training data. Return history of loss and accuracy for each epoch
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
    
# Evaluate the model with the test data to get the scores on 'real' data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# Plot data to see relationships in training and validation data
import numpy as np
import matplotlib.pyplot as plt
epoch_list = list(range(1, len(history.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, history.history['acc'], epoch_list, history.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
