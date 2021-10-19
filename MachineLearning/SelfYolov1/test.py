import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

a = np.array([[1],
			  [2],
			  [3],
			  [4]])

print((K.shape(a)[0],) + (1,1,2))


class Yolo_Reshape(tf.keras.layers.Layer):
	def __init__(self, target_shape):
		super(Yolo_Reshape).__init__()
		self.targer_shape = target_shape

	def get_config(self):
		config = supee().get_config().copy()
		config.update({'targer_shape':self.targer_shape})
		return config

	def call(self, input):