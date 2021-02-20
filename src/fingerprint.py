# File for Base class to store fingerprint
from . import util

import cv2 as cv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class fp:
	def __init__(self, path_fp_img, segment_block_size):
		self.original_img = cv.imread(path_fp_img, cv.IMREAD_GRAYSCALE)
		# print(self.original_img)

		# Preprocessing
		# Segmentation
		# Generate segmentation mask
		global_greyscale_variance = np.var(self.original_img)*0.07
		self.segmentation_mask = np.ones(self.original_img.shape)
		x = 0
		while True:
			y = 0
			while True:
				local_grayscale_variance = np.var(self.original_img[x: x+segment_block_size, y: y+segment_block_size])
				if local_grayscale_variance <= global_greyscale_variance:
					self.segmentation_mask[x: x+segment_block_size, y: y+segment_block_size] = 0
				y += segment_block_size
				if y >= self.original_img.shape[1]:
					break
			x += segment_block_size
			if x >= self.original_img.shape[0]:
				break
		
		# Apply Segmentation mask
		self.segmented_image = self.original_img.copy()
		kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(2*segment_block_size, 2*segment_block_size))
		self.segmentation_mask = cv.morphologyEx(self.segmentation_mask, cv.MORPH_OPEN, kernel_open_close)
		self.segmentation_mask = cv.morphologyEx(self.segmentation_mask, cv.MORPH_CLOSE, kernel_open_close)
		self.segmented_image[self.segmentation_mask == 0] = 255

		# Normalization
		required_variance = 4096.0
		required_mean = 127.0
		current_variance = np.var(self.segmented_image)
		current_mean = np.mean(self.segmented_image)
		temp_dev = ((required_variance/current_variance)*((self.segmented_image-current_mean)**2)) ** 0.5
		self.normalized_image = np.where(self.segmented_image > current_mean, required_mean + temp_dev, required_mean - temp_dev)

		cv.imshow("normalized image", self.normalized_image)
		cv.waitKey(0)
