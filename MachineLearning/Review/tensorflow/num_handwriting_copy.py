import numpy as np
import cv2
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_val, Y_val = X_train[50000:60000, :], Y_train[50000:60000]
X_train, Y_train = X_train[:50000, :], Y_train[:50000]

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
X_val = X_val.reshape(X_val.shape[0], 28,28,1)

#Onehot coding

Y_train = tf.keras.utils.to_categorical(Y_train, 10)

Y_test = tf.keras.utils.to_categorical(Y_test, 10)

Y_val = tf.keras.utils.to_categorical(Y_val,10)
#end data preparation

#define model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='sigmoid', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=10, validation_data =(X_val, Y_val))

model.save("num.h5")
#test

cv2.imshow("test", X_test[19])

predict_y = model.predict(X_test[19].reshape(1,28,28,1))
print("Predicted value: ", np.argmax(predict_y))
cv2.waitKey(0)

