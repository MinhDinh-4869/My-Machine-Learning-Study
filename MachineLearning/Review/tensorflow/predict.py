import numpy as np
import tensorflow as tf
import cv2


#model = tf.keras.models.load_model("./model/Mymodel.keras")
model = tf.keras.models.load_model('MYMODEL')
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], 28,28,1)

cv2.imshow("Img", X_test[12].reshape(28,28))
cv2.waitKey(0)
y_predict = model.predict(X_test[12].reshape(1,28,28,1))
print("Predict value: ", np.argmax(y_predict))
