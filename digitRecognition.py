## Keras is used to build a Convolutional Neural Network
# with the same architecture proposed as an example in the Keras library
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# 
# Some preprocessing steps are added, which I learned from these 2 sources:
# 1/ https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
# 2/ https://www.youtube.com/watch?v=j-3vuBynnOE&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2 

# Use print as a function
from __future__ import print_function
import keras
# Simple list of layers (single-input, single-output)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy
import os
from cv2 import cv2
import random
# For multi-dimentional image processing
from scipy import ndimage

# To centralize the image according to its center of mass
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    # Returns tuple of number of rows & columns
    rows, cols = img.shape
    shiftx = numpy.round(cols/2.0-cx).astype(int)
    shifty = numpy.round(rows/2.0-cy).astype(int)

    return shiftx, shifty

# To Shift the image in the given directions
def shift(img, sx, sy):
    rows, cols = img.shape
    M = numpy.float32([[1,0,sx],[0,1,sy]])
    # Translation (shift the object location)
    shifted = cv2.warpAffine(img, M, (cols,rows))
    return shifted

# We will do the same centralization for Sudoku Digit
# according to their center of mass, so that all digit image is in SAME form
def shift_acc_center_of_mass(img):
    img = cv2.bitwise_not(img)

    # centralize according to center of mass
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img

# Number of samples
batch_size = 128
# 9 for digits 1-9 for the sudoku
num_classes = 9
# Number of times you go through the training set
epochs = 35

# Image dimensions
img_rows, img_cols = 28, 28

# The path in my system where data is stored
##--- change this if you want to train data model on your own ---##
DATADIR = "D:/RTSudoku"
CATEGORIES = ["1","2","3","4","5","6","7","8","9"]

# Read training data (Stored in folders named 1,2..,9 in same directory)
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_rows, img_cols))
            new_array = shift_acc_center_of_mass(new_array)
            training_data.append([new_array, class_num])

create_training_data()

# Mix the data
random.shuffle(training_data)

# Splitting the data 80-20
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(training_data)*8//10):
    x_train.append(training_data[i][0])
    y_train.append(training_data[i][1])
for i in range(len(training_data)*8//10, len(training_data)):
    x_test.append(training_data[i][0])
    y_test.append(training_data[i][1])

# Reshape
x_train = numpy.array(x_train)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = numpy.array(x_test)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data
x_train = x_train / 255
x_test = x_train / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Group a linear stack of layers
model = Sequential()
# 2D Convolution because of image, input_shape is the dimentions of image
# Rectified linear unit to return positive values, zero if negative
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
# Downsampling with max operation on stride of size 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Randomly set input units as 0 with 0.25 frequency, done to avoid overfitting
model.add(Dropout(0.25))
# Flatten the input
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# Randomly set input units as 0 with 0.5 frequency, done to avoid overfitting
model.add(Dropout(0.5))
# softmax outputs a vector that represent probability distribution of potential outcomes
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Saving the weights in this file
# h5 file extension is used to save multidimensional arrays of data
##--- Note- You don't need to train the CNN on your own computer ---##
##--- Keep this file in your project directory ---##
model.save_weights('digitRecognition.h5')