import tensorflow as tf 
import numpy as np
import cv2


(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

model = tf.keras.models.load_model("myModel.h5")

for i in range(20):
	idx = np.random.randint(0,10000)
	cv2.imshow("img", X_test[idx])
	y_pred = model.predict(X_test[idx].reshape(1,28,28,1))
	print("Predicted value: ", np.argmax(y_pred))
	cv2.waitKey(0)