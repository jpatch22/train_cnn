#!/usr/bin/env python
import numpy as np
import cv2
import os
import string
from PIL import Image, ImageFont, ImageDraw

class Data_Gen_Num:
	def __init__(self):
		files = os.listdir("/home/fizzer/train_cnn/lp_letter_train_data/adjusted_data")
		for file in files:
			os.remove("/home/fizzer/train_cnn/lp_letter_train_data/adjusted_data" + file)

		collected_images_labels = os.listdir("/home/fizzer/train_cnn/lp_letter_train_data/collected_data")
		self.images = []
		for cil in collected_images_labels:
			self.images.append((cil[0], cv2.imread("/home/fizzer/train_cnn/lp_letter_train_data/collected_data/" + cil)))
		print(len(self.images))

	def copy(self):
		for tag, image in self.images:
			cv2.imwrite(os.path.join("/home/fizzer/train_cnn/lp_letter_train_data/adjusted_data/", 
		                             "{}.png".format(tag)),
		                             image)

	def generate_blur(self, kernel_size):
		for tag, image in self.images:
			blurred = cv2.blur(image, (kernel_size, kernel_size))
			cv2.imwrite(os.path.join("/home/fizzer/train_cnn/lp_letter_train_data/adjusted_data/", 
		                             "{}_blurSize_{}.png".format(tag, kernel_size)),
		                             blurred)


	def generate_rotated_images(self, angle):
		for tag, image in self.images:
			rotated = self.rotate_image(image, angle)
			cv2.imwrite(os.path.join("/home/fizzer/train_cnn/lp_letter_train_data/adjusted_data/", 
		                             "{}_rotated_{}.png".format(tag, angle)),
		                             rotated)

	def rotate_image(self, image, angle):
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
		return result


x = Data_Gen_Num()
x.copy()
for i in range(1, 20):
	x.generate_blur(i)
for i in range(-20, 20, 1):
	x.generate_rotated_images(i)