import numpy as np
import tensorflow as tf
import tf.keras.backend as K

#Documents
#https://www.maskaravivek.com/post/yolov1/
#https://www.tensorflow.org/guide/keras/custom_layers_and_models
#https://stackoverflow.com/questions/61560888/keras-dense-layer-output-shape


#Create a custom layer for the output layer, as tensorflow does not support reshape as we want to.
#As the layers in tensorflow are classes, we also have to define a class

#constructor
#def __init__(self, target_shape):
#	super(Yolo_Reshape).__init__()
#   self.target_shape = tuple(target_shape)

#get config
#def get_config(self):
#	config = super().get_config().copy()
#	config.update({'target_shape':self.target_shape})
#	return config

#def call(self, input=(,1470)_size_vector):
#	define S, B, C
#	input structure: [<classes><confidences><boxes>]
#	define indices to cut: idx1 = 7x7x20 as the classes appear first
#						   idx2 = idx1 + 7x7x2 as the confidences appear next
#	define class_probs, confidences, boxes:
#	K.reshape(input[:, something], (K.shape(input)[0],) + (7, 7, 20/2/2*4) )
# 										 ^                     ^
#										 |                     |
#				tensor infor, formatting by tensorflow      shape u want to reshape
#	output = K.concatenate([class_probs, confidences, boxes])
# return output


class Yolo_Reshape_Layer(tf.keras.layers.Layer):
	def __init__(self, target_shape):
		#constructor, take the shape we want as input parameter.
		super(Yolo_Reshape_Layer, self).__init__()
		self.target_shape = tuple(target_shape)

	#idk but this must be included when creating a new layer.
	
	def get_config(self):
		config = super().get_config().copy() #create a config(?)
		#using config(?) to update states.
		config.update({
			'target_shape': self.target_shape
			})
		return config

	#This method is required. same reason to get_config()
	#About input in tensorflow:
	#Shape:
	#(num_of_tensor, __tensors__)
	#Matrix which rows are tensors(?)

	def call(self, input_tensor):
		#grid 7x7
		S = [self.target_shape[0], self.target_shape[1]] #target_shape = 7 x 7 x 30

		#classes
		C = 20

		# #bounding boxes per grid
		B = 2

		#As the output of tensorflow Dense layer is 7x7x30- length vector, we have to
		#determine which point separate the confidences, boxes' size, and classes prediction

		idx1 = S[0]*S[1]*C #classes prediction 7x7x20
		idx2 = idx1 + S[0]*S[1]*B  #confidences 7x7x2


		#reshape: tf.keras.backend.reshape(x, shape) 
		#input shape: Dense layer -> Nx1 column vector
		#
		#CLASS PROBABILITY
		class_probs = K.reshape(input_tensor[:, :idx1], (K.shape(input_tensor)[0],) + tuple(S[0], S[1], C))
		class_probs = K.softmax(class_probs)
		#result -> something like this: (<tf.Tensor 'strided_slice:0' shape=() dtype=int32>, 7, 7, 20)


		#As the input has shape: (,1470) <==> <classes..., confidences..., boxes...>
		#from tf.keras.layers.Dense(class_num)
		#shape Nx1

		#same
		#confidence
		confidence = K.shape(input_tensor[:, idx1:idx2], (K.shape(input_tensor)[0],) + tuple(S[0], S[1], B)) 
		confidence = K.sigmoid(confidence)

		#boxes
		boxes = K.shape(input_tensor[:,idx2: ], (K.shape(input_tensor[0]),) + tuple(S[0], S[1], 4*B)) 
		boxes = K.sigmoid(boxes)

		output = K.concatenate([class_probs, confidence, boxes])
		#output like:
		#                   \______________\_\_________\
		#					||	           | |         |
		#					|| class_probs |c|  boxes  |
		#					 \_____________|_|_________|
		return output

class Yolo_Reshape(tf.keras.layers.Layer):
	def __init__(self, target_shape):
		super(Yolo_Reshape).__init__()
		self.target_shape = tuple(target_shape)

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'target_shape': self.target_shape
			})
		return config

	def call(self, input):
		#grid
		S = [self.target_shape[0], self.target_shape[1]]

		#classes
		C = 20

		#boxes
		B = 2



		idx1 = S[0]*S[1]*C
		idx2 = idx1 + S[0]*S[1]*B
		#input = [classes, confidences, boxes info]
		#reshape
		
		class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C])) 
		#result -> something like this: (<tf.Tensor 'strided_slice:0' shape=() dtype=int32>, 7, 7, 20)
		class_probs = K.softmax(class_probs)

		#confs
		confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
		confs - K.sigmoid(confs)

		#boxes
		boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], 4*B]))
		boxes = K.sigmoid(boxes)

		#out
		outputs = K.concatenate([class_probs, confs, boxes])
		return outputs

class Yolo_Reshape_copy(tf.keras.layers.Layer):
	def __init__(self, target_shape):
		super(Yolo_Reshape_copy).__init__()
		self.target_shape = tuple(target_shape)

	def get_config(self):
		config = super().get_config().copy()
		config.update({'target_shape': self.target_shape})
		return config

	def call(self, input):
		#grid
		S = [self.target_shape[0], self.target_shape[1]]
		C = 20

		#data position: 
		#[classes..., confs..., boxes...] -->(,1470)
		#Dense layer -> (none, nodes_num)
		idx1 = S[0]*S[1]*C
		idx2 = idx1 + S[0]*S[1]*B

		class_probs = K.reshape(input[:, :idx1], (K.shape(input[0]),) + tuple([S[0], S[1], C]))
		class_probs = K.softmax(class_probs)

		confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + (tuple([S[0], S[1], B])))
		confs = K.sigmoid(confs)

		boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], 4*B]))
		boxes = K.sigmoid(boxes)

		outputs = K.concatenate([class_probs, confs, boxes])
		#output like:
		#                   ___________________________
		#                   \______________\_\_________\
		#					||	           | |         |
		#					|| class_probs |c|  boxes  |
		#					 \_____________|_|_________|
		return outputs

class Yolo_Reshape_1(tf.keras.layers.Layer):
	def __init__(self, target_shape):
		super(Yolo_Reshape_1).__init__()
		self.target_shape = tuple(target_shape)

	def get_config(self):
		config = super().get_config().copy()
		config.update({'target_shape': self.target_shape})
		return config
	def call(self, input):
		#input = vector (,1470)
		# Grid 
		S = [self.target_shape[0], self.target_shape[1]] #-> (7,7,30)
		# classes:
		C = 20
		# boxes
		B = 2

		#input format: (,1470) vector: <20 classes><B confidences><4*B boxes info>
		idx1 = S[0]*S[1]*C
		idx2 = idx1 + S[0]*S[1]*B

		#class probs
		#K.reshape(x, size)
		class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C])) # reshape to 7x7x20
		class_probs = K.softmax(class_probs)

		#confidences
		confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) +tuple([S[0], S[1], B])) # reshape to 7x7x2
		confs = K.sigmoid(confs)

		#boxes
		boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], 4*B])) #-->reshape to 7x7x4*B
		boxes = K.sigmoid(boxes)

		outputs = K.concatenate([class_probs, confs, boxes])

		return outputs
def yolo_model():
	model = tf.keras.models.Sequential()

	#add layers
	model.add(tf.keras.layers.Conv2D(64, (7,7),  padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(448,448,3)))
	model.add(tf.keras.layers.MaxPooling2D((2,2), strides= (2,2)))

	model.add(tf.keras.layers.Conv2D(192, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(tf.keras.layers.Conv2D(128, (1,1), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(tf.keras.layers.Conv2D(256, (1,1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(256, (1,1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(256, (1,1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(256, (1,1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))			
	model.add(tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(1024, (3,3), strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))

	model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1)))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1407, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))

	model.add(Yolo_Reshape_Layer(target_shape=(7,7,30)))

	return model
model = yolo_model()
model.summary()