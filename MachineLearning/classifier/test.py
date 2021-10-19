import tensorflow as tf 
import numpy as np 
import cv2

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()


#look at the shape to put :, like:
#(60000, 28, 28) ->60000 -> number of frames, then put (begin:end,: )
#(,1470) --> (:, begin:end)

X_val, Y_val = X_train[50000:60000, :], Y_train[50000:60000]
X_train, Y_train = X_train[:50000, :], Y_train[:50000]


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28 ,28, 1)

#one hot

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)
Y_val = tf.keras.utils.to_categorical(Y_val, 10)

#define model

inputs = tf.keras.Input(shape=(28, 28, 1))

conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid')(inputs)
pool1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv1)
conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv2)

flatt = tf.keras.layers.Flatten()(pool2)
dense1 = tf.keras.layers.Dense(128, activation='sigmoid')(flatt)
dense2 = tf.keras.layers.Dense(10, activation='softmax')(dense1)

classifier = tf.keras.Model(inputs= inputs, outputs= dense2)

classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, Y_train, epochs=10, validation_data = (X_val, Y_val))

classifier.save("myModel.h5")

cv2.imshow("afs", X_test[12])

ypred = classifier.predict(X_test[12].reshape(1,28,28,1))
print("predicted val: ", np.argmax(ypred))
cv2.waitKey(0)