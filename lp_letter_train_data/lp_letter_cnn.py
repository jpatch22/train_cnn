#!/usr/bin/env python
import numpy as np
import cv2
import os
import string
from PIL import Image, ImageFont, ImageDraw
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

class CNN:
	def __init__(self):
		pass

	def convert_to_array(self, l):
		all_chars_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
		index = all_chars_list.index(l)
		label = np.zeros(26)
		label[index] = 1
		return label

	def create_data_sets(self):
		path = "/home/fizzer/train_cnn/lp_letter_train_data/adjusted_data"
		labels = os.listdir(path)
		
		X_dataset = []
		Y_dataset = []
		
		for i in range(0, len(labels)):
			label = labels[i][0]
			# print(label)
			image = cv2.imread(path + "/" + labels[i])
			y = self.convert_to_array(label)
			Y_dataset.append(y)
			x = np.array(image)
			X_dataset.append(x)
			# print(x.shape)


		X_dataset = np.array(X_dataset)
		Y_dataset = np.array(Y_dataset)

		return X_dataset, Y_dataset

	def create_val_data(self):
		path = "/home/fizzer/train_cnn/lp_letter_train_data/val_data"
		labels = os.listdir(path)
		
		X_dataset = []
		Y_dataset = []
		
		for i in range(0, len(labels)):
			label = labels[i][0]
			print(label)
			image = cv2.imread(path + "/" + labels[i])
			y = self.convert_to_array(label)
			Y_dataset.append(y)
			x = np.array(image)
			X_dataset.append(x)
			# print(x.shape)



		X_dataset = np.array(X_dataset)
		Y_dataset = np.array(Y_dataset)

		return X_dataset, Y_dataset


	def neural_net(self, X_train_dataset, Y_train_dataset, X_val_dataset, Y_val_dataset):

		conv_model = models.Sequential()
		conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
		                             input_shape=(140, 140, 3)))
		conv_model.add(layers.MaxPooling2D((2, 2)))
		conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		conv_model.add(layers.MaxPooling2D((2, 2)))
		conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
		conv_model.add(layers.MaxPooling2D((2, 2)))
		conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
		conv_model.add(layers.MaxPooling2D((2, 2)))
		conv_model.add(layers.Flatten())
		conv_model.add(layers.Dropout(0.5))
		conv_model.add(layers.Dense(512, activation='relu'))
		conv_model.add(layers.Dense(26, activation='softmax'))

		LEARNING_RATE = 1e-4
		conv_model.compile(loss='categorical_crossentropy',
		                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
		                   metrics=['acc'])

		self.reset_weights(conv_model)
		history_conv = conv_model.fit(X_train_dataset, Y_train_dataset, 
                              validation_data=(X_val_dataset, Y_val_dataset), 
                              epochs=15, 
                              batch_size=16)
		print(history_conv)
		conv_model.save('/home/fizzer/train_cnn/lp_letter_train_data/lp_letter_model')

	def reset_weights(self, model):
	    session = backend.get_session()
	    for layer in model.layers: 
	        if hasattr(layer, 'kernel_initializer'):
	            layer.kernel.initializer.run(session=session)



c = CNN()
x_train, y_train = c.create_data_sets()
x_val, y_val = c.create_val_data()
c.neural_net(x_train, y_train, x_val, y_val)