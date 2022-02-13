import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix, classification_report

# Loading Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# One Hot Encoding
y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_train, num_classes=10)

x_train = cv2.normalize(x_train, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
x_test = cv2.normalize(x_test, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(4, 4), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_cat_train, epochs=50, verbose=2)
evaluation = model.evaluate(x_test, y_cat_test)
predictions = model.predict_classes(x_test)
print("Evaluation: ", evaluation)
print(classification_report(y_test, predictions))
