import numpy as np
import cv2
import tensorflow as tf

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

#cut 
X_val, Y_val = X_train[50000:60000, :], Y_train[50000:60000]
X_train, Y_train = X_train[:50000, :], Y_train[:50000]

#one_hot coding

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_val = tf.keras.utils.to_categorical(Y_val, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

#reshape X

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)


#define model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid', input_shape =(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
history = model.fit(X_train, Y_train, epochs=10, validation_data = (X_val, Y_val))

model.save("cloth.h5")

cv2.imshow("cloth", X_test[12])
cv2.waitKey(0)
pre = model.predict(X_test[12].reshape(1,28,28,1))
class_name = class_names[int(argmax(pre))]
print(class_name) 