import numpy as np 
import tensorflow as tf
import tensorflow.keras.backend as K

class Yolo_Reshape(tf.keras.layers.Layer):
	def __init__(self, target_shape):
		super(Yolo_Reshape).__init__()
		self.target_shape = tuple(target_shape)

	def get_config(self):
		config = super().get_config().copy()
		config.update({'target_shape':self.target_shape})
		return config

	def call(self, input):
		#grid
		S = [self.target_shape[0], self.target_shape[1]] #--> target_shape = (7, 7, 30)

		#boxes
		B = 2

		#classes
		C = 20

		# input  = <classes><confidence><boxes>

		idx1 = S[0]*S[1]*C
		idx2 = idx1 + S[0]*S[1]*B

		#input shape = (,1470)

		class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C])) #tuple + tuple
		class_probs = K.softmax(class_probs)

		confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
		confs = K.sigmoid(confs)

		boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], 4*B]))
		boxes = K.sigmoid(boxes)

		output = K.concatenate([class_probs, confs, boxes])
		return output


def softmax_regression(X_train, Y_train, epochs, eta):
	w_init = np.random.random((C, X_train.shape[1]))
	W = [w_init]
	check_after = 20
	count = 0
	for i in range(epochs):
		x_index = np.random.permutation(X_train.shape[0])
		for j in x_index:
			xi = X_train[j, :].reshape(1, X_train.shape[1])
			yi = Y_train[j, :].reshape(1, C)

			zi = xi.dot(W[-1].T)
			ai = softmax(zi[0]).reshape(1, C)

			dW = eta*(ai - yi).T.dot(xi)
			w_new = W[-1] - dW
			count+=1
			if(count % check_after == 0):
				if(np.linalg.norm(w_new - W[-check_after]) < 1e+4):
					return W
			W.append(w_new)
	return W
