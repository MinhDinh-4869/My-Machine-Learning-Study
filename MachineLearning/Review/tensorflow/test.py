import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

#print(X_train.shape)

X_valid, Y_valid = X_train[50000:60000, :], Y_train[50000:60000]
X_train, Y_train = X_train[:50000, :], Y_train[:50000]


#one-hot coding

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_valid = tf.keras.utils.to_categorical(Y_valid, 10)
Y_test  = tf.keras.utils.to_categorical(Y_test,  10)

#reshape data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #input is tensor(28,28,1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
X_test  = X_test.reshape(X_test.shape[0],   28, 28, 1)

#define model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'sigmoid', input_shape = (28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #output type -> classifier

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, Y_train, epochs = 10, validation_data = (X_valid, Y_valid))

model.save("Model.pb")


cv2.imshow("xtest",X_test[1112])
y_predict = model.predict(X_test[1112].reshape(1,28,28,1))
print("predicted value: ", argmax(y_predict))