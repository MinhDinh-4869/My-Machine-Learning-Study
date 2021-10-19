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

model = tf.keras.models.load_model("cloth.h5")


cv2.imshow("cloth", X_test[1212])


for i in range(10):
    idx = np.random.randint(0,10000)
    cv2.imshow("cloth", X_test[idx])
    pre = model.predict(X_test[idx].reshape(1,28,28,1))
    class_name_pre = class_names[int(np.argmax(pre))]
    class_name_true = class_names[int(np.argmax(Y_test[idx]))]
    print("predicted class: ", class_name_pre, end="\t")
    print("real class: ", class_name_true) 
    cv2.waitKey(0)