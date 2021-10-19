import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


#load data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_validation, Y_validation = X_train[50000:60000,  :], Y_train[50000:60000]
X_train, Y_train  = X_train[:50000, :], Y_train[:50000]
#one-hot coding labels

#Y_train = tf.keras.utils.to_categorical(Y_train, 10)
#Y_validation = tf.keras.utils.to_categorical(Y_validation, 10)
#Y_test = tf.keras.utils.to_categorical(Y_test, 10)


#Y_train = tf.keras.utils.np_utils.to_catigorical(Y_train,10)
#Y_test = tf.keras.utils.np_utils.to_categorical(Y_test, 10)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1)

#define model
#model is an object of class Sequential
model = tf.keras.models.Sequential() #from models create an object of Sequential class

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid', input_shape = (28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compile model

#model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#NOTE: 
#using loss = 'categorical_crossentropy' requiring labels to be in one-hot coding type
#			  'sparse_categorical_crossentropy' do not require one-hot coding
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_validation, Y_validation))
model.save("MYMODEL")

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_validation,Y_validation  , verbose=2)



cv2.imshow("img", X_test[122])
y_predict = model.predict(X_test[122].reshape(1,28,28,1))
class_pre = np.argmax(y_predict)
print("predicted value: ", class_pre)
cv2.waitKey(0)

