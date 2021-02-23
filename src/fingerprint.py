# File for Base class to store fingerprint
from . import util

import cv2 as cv
import sys
import numpy as np
from skimage.draw import line
from math import floor, ceil
np.set_printoptions(threshold=sys.maxsize)

class fp:
	def __init__(self, path_fp_img, segment_block_size):
		self.original_img = cv.imread(path_fp_img, cv.IMREAD_GRAYSCALE)
		# kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(segment_block_size//10, segment_block_size//10))
		# self.original_img = cv.morphologyEx(self.original_img, cv.MORPH_OPEN, kernel_open_close)
		# self.original_img = cv.morphologyEx(self.original_img, cv.MORPH_CLOSE, kernel_open_close)
		self.segment_block_size = segment_block_size

		# Preprocessing
		# Segmentation
		# Generate segmentation mask
		# TODO make efficient
		global_greyscale_variance = np.var(self.original_img)*0.1
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
		# Opening Closing to remove specks
		# kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(2*segment_block_size, 2*segment_block_size))
		# self.segmentation_mask = cv.morphologyEx(self.segmentation_mask, cv.MORPH_OPEN, kernel_open_close)
		# self.segmentation_mask = cv.morphologyEx(self.segmentation_mask, cv.MORPH_CLOSE, kernel_open_close)
		self.segmented_image[self.segmentation_mask == 0] = 255

		# Normalization
		required_variance = 15000.0
		required_mean = 127.0
		current_variance = np.var(self.segmented_image)
		current_mean = np.mean(self.segmented_image)
		temp_dev = ((required_variance/current_variance)*((self.segmented_image-current_mean)**2)) ** 0.5
		self.normalized_image = np.where(self.segmented_image > current_mean, required_mean + temp_dev, required_mean - temp_dev)

		# Ridge orientation
		grad_x = cv.Sobel(self.normalized_image, cv.CV_64F, 1, 0, ksize=3)
		grad_y = cv.Sobel(self.normalized_image, cv.CV_64F, 0, 1, ksize=3)

		local_directions_x = np.zeros(self.original_img.shape)
		local_directions_y = np.zeros(self.original_img.shape)
		for i in range(self.original_img.shape[0]):
			for j in range(self.original_img.shape[1]):
				start_i = max(0, i-segment_block_size//2)
				end_i = min(i+segment_block_size//2, self.original_img.shape[0])
				start_j = max(0, j-segment_block_size//2)
				end_j = min(j+segment_block_size//2, self.original_img.shape[1])
				local_directions_x[i, j] = np.sum(2*grad_x[start_i: end_i, start_j: end_j]*grad_y[start_i: end_i, start_j: end_j])
				local_directions_y[i, j] = np.sum(grad_x[start_i: end_i, start_j: end_j]**2-grad_y[start_i: end_i, start_j: end_j]**2)
		
		gaussian_blur_kernel_size = (2*segment_block_size+1, 2*segment_block_size+1)
		gaussian_std = 1.0
		gaussian_local_directions_x = cv.GaussianBlur(local_directions_x, gaussian_blur_kernel_size, gaussian_std)
		gaussian_local_directions_y = cv.GaussianBlur(local_directions_y, gaussian_blur_kernel_size, gaussian_std)

		self.orientation_map = 0.5*(np.arctan2(gaussian_local_directions_x, gaussian_local_directions_y)+np.pi)

		self.frequency_map = np.zeros(self.original_img.shape)
		buffer_windows = int(ceil(segment_block_size / 10))
		for i in range(0, self.original_img.shape[0], segment_block_size):
			for j in range(0, self.original_img.shape[1], segment_block_size):
				end_i = min(self.original_img.shape[0], i+segment_block_size)
				end_j = min(self.original_img.shape[1], j+segment_block_size)
				line_direction = np.average(self.orientation_map[i:end_i, j:end_j])
				middle_line = None
				if np.pi / 4 < line_direction < 3 * np.pi / 4:
					mid_j = (j+end_j)/2
					diff_i = end_i - i
					offset_j = diff_i / 2 / np.tan(line_direction)
					middle_line = zip(*line(i, floor(mid_j+offset_j), end_i, floor(mid_j-offset_j)))
				else:
					mid_i = (i+end_i)/2
					diff_j = end_j - j
					offset_i = np.tan(line_direction) * diff_j/2
					middle_line = zip(*line(floor(mid_i+offset_i), j, floor(mid_i-offset_i), end_j))

				is_ridge = None
				def check_ridge(x, y): 
					if self.normalized_image[x, y] > required_mean:
						return True
					return False
				
				num_changes = 0
				current_buffer_window = 0
				for x, y in middle_line:
					try:
						if is_ridge is None:
							is_ridge = check_ridge(x, y)
						else:
							current_check = check_ridge(x, y)
							current_buffer_window += 1
							if is_ridge != current_check and current_buffer_window <= buffer_windows:
								num_changes+=1
								is_ridge = current_check
								current_buffer_window = 0
					except IndexError:
						continue

				self.frequency_map[i:end_i, j:end_j] = ceil(num_changes/2)
		
		cv.imshow("seg mask", self.segmentation_mask)
		cv.waitKey(0)
		cv.imshow("norm", self.normalized_image)
		cv.waitKey(0)
		cv.imshow("orient", self.orientation_map/np.pi)
		cv.waitKey(0)
		cv.imshow("freq", self.frequency_map/3.0)
		cv.waitKey(0)

