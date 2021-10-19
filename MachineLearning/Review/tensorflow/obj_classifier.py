import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

X_valid, Y_valid = X_train[50000:60000, :], Y_train[50000:60000]
X_train, Y_train = X_train[:50000,:], Y_train[:50000]

for i in range(10):
	cv2.imshow(class_names[int(Y_train[i])], X_train.reshape(28,28,1))

cv2.waitKey(0)

#one hot coding

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_valid = tf.keras.utils.to_categorical(Y_valid, 10)
Y_test  = tf.keras.utils.to_categorical(Y_test, 10)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#define model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'sigmoid', input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'sigmoid'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, Y_train, epochs = 10, validation_data = (X_valid, Y_valid))
model.save("MYMODEL_2.h5")

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_valid,Y_valid  , verbose=2)



cv2.imshow("img", X_test[122])
y_predict = model.predict(X_test[122].reshape(1,28,28,1))
class_pre = np.argmax(y_predict)
print("predicted value: ", class_names[int(class_pre)])
cv2.waitKey(0)

