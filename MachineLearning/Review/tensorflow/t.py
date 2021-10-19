import tensorflow as tf
import numpy as np

(xt, yt), (xte, yte) = tf.keras.datasets.mnist.load_data()

y = yt[:10]

y = tf.keras.utils.to_categorical(y, 10)


#create a Sequential object
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid', input_shape=(28,28,1)))