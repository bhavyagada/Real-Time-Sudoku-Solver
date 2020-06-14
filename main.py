##----- Entry point, Run this file!! -----##

# NOTE: I was new to all the Machine Learning and Image Processing when I built this project.
# Most of my ideas and codes were learned from others.

# To see the list of all the resources, see the README.txt

# No need of running the digitRecognition.py file to train the Convolutional Neural Network (CNN).
# The trained architecture is saved in digitRecognition.h5


#---------- IMPORTS ----------#

# For Analysis of Images
import numpy as np
# For Image Processing
from cv2 import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import RTSudokuSolver

##---------- START VIDEO AND PRINT SOLVED IMAGE ----------##

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

# Load and set up camera
#For Windows WebCams are exclusively handeled by DirectShow API
#700 is the flag for CAP_DSHOW (DirectShow)
cap = cv2.VideoCapture(0, 700)

# HD Camera
cap.set(3, 1280)
cap.set(4, 720)

# Define codec to save video
# out = cv2.VideoWriter("solution_video.avi", cv2.VideoWriter_fourcc('X','V','I','D'), 20.0, (640,480))

# Loading model (Loading weights and configuration seperately to speed up model.predict)
input_shape = (28, 28, 1)
num_classes = 9
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Load pre-trained model weights. (Model trained in digitRecognition.py)
model.load_weights("digitRecognition.h5")

old_sudoku = None
while True:
    # Read the frame
    ret, frame = cap.read()

    if ret == True:
        sudoku_frame = RTSudokuSolver.recognize_and_solve(frame, model, old_sudoku)
        
        # Print solved image
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)
        
        # Save the video
        # comment this line if you don't want to save video
        # out.write(frame)
        
        # waitKey(1) for continuously capturing images and quit by pressing 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release unneeded resources
cap.release()
# out.release()
cv2.destroyAllWindows()