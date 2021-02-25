# File for Base class to store fingerprint
from . import util

import cv2 as cv
import sys
import numpy as np
from math import floor, ceil
from skimage.draw import line
from skimage.morphology import skeletonize
import fingerprint_enhancer
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
		self.segmentation_mask = np.ones(self.original_img.shape)
		global_greyscale_variance = np.var(self.original_img)*0.1
		for i in range(0, self.original_img.shape[0], segment_block_size):
			for j in range(0, self.original_img.shape[1], segment_block_size):
				end_i = min(self.original_img.shape[0], i+segment_block_size)
				end_j = min(self.original_img.shape[1], j+segment_block_size)
				thinned_local_grayscale_variance = np.var(self.original_img[i: end_i, j: end_j])
				if thinned_local_grayscale_variance <= global_greyscale_variance:
					self.segmentation_mask[i: end_i, j: end_j] = 0
		
		# Apply Segmentation mask
		self.segmented_image = self.original_img.copy()
		# Opening Closing to remove specks
		kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(2*segment_block_size, 2*segment_block_size))
		self.segmentation_mask = cv.morphologyEx(self.segmentation_mask, cv.MORPH_CLOSE, kernel_open_close)
		self.segmentation_mask = cv.morphologyEx(self.segmentation_mask, cv.MORPH_OPEN, kernel_open_close)
		self.segmented_image[self.segmentation_mask == 0] = 255

		# Normalization
		required_variance = 50000.0
		required_mean = 127.0
		current_variance = np.var(self.segmented_image)
		current_mean = np.mean(self.segmented_image)
		temp_dev = ((required_variance/current_variance)*((self.segmented_image-current_mean)**2)) ** 0.5
		self.normalized_image = np.where(self.segmented_image > current_mean, required_mean + temp_dev, required_mean - temp_dev)

		# Ridge orientation calculation
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

		# self.orientation_map = 0.5*(np.arctan2(gaussian_local_directions_x, gaussian_local_directions_y)+np.pi)
		self.orientation_map = 0.5*(np.arctan2(gaussian_local_directions_x, gaussian_local_directions_y)+np.pi)

		# Frequency map calculation
		self.frequency_map = np.zeros(self.original_img.shape)
		buffer_windows = int(ceil(segment_block_size / 5))
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
				
				ridges = []
				current_buffer_window = 0
				for x, y in middle_line:
					try:
						if is_ridge is None:
							is_ridge = check_ridge(x, y)
							if is_ridge:
								ridges.append((x, y))
						else:
							current_check = check_ridge(x, y)
							current_buffer_window += 1
							# if is_ridge != current_check and current_buffer_window <= buffer_windows:
							if is_ridge != current_check:
								is_ridge = current_check
								current_buffer_window = 0
								if is_ridge:
									ridges.append((x, y))
					except IndexError:
						continue
				
				distance_sum = 0.0
				for i in range(len(ridges)-1):
					distance_sum += util.euclidean_distance(*ridges[i], *ridges[i+1])
				if len(ridges) == 1:
					distance_sum = segment_block_size

				local_freq = None
				if distance_sum == 0:
					local_freq = 1.0
				else:
					local_freq = len(ridges)/distance_sum

				self.frequency_map[i:end_i, j:end_j] = local_freq
		
		# Apply Gabor filter
		self.enhanced_image = np.zeros(self.original_img.shape)
		for i in range(0, self.original_img.shape[0], segment_block_size):
			for j in range(0, self.original_img.shape[1], segment_block_size):
				end_i = min(self.original_img.shape[0], i+segment_block_size)
				end_j = min(self.original_img.shape[1], j+segment_block_size)
				average_orientation = np.average(self.orientation_map[i:end_i, j:end_j]) + np.pi
				gabor_kernel = cv.getGaborKernel((segment_block_size, segment_block_size), 5*self.frequency_map[i, j], average_orientation, 10.0*self.frequency_map[i, j], 1.0)
				# gabor_kernel = cv.getGaborKernel((segment_block_size, segment_block_size), 5.0, average_orientation, 1.0, 1.0)
				self.enhanced_image[i: end_i, j: end_j] = cv.filter2D(self.normalized_image[i: end_i, j: end_j], cv.CV_64F, gabor_kernel)
		
		self.enhanced_image = fingerprint_enhancer.enhance_Fingerprint(self.normalized_image)

		# self.binary_image, _ = cv.threshold(self.enhanced_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

		# image_to_thin = 255.0 - self.binary_image
		# image_to_thin = 255.0 - self.enhanced_image
		# thinning_kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
		# self.thinned_image = np.zeros(self.original_img.shape)
 
		# while (cv.countNonZero(image_to_thin)!=0):
		# 	erode = cv.erode(image_to_thin,thinning_kernel)
		# 	opening = cv.morphologyEx(erode,cv.MORPH_OPEN,thinning_kernel)
		# 	subset = erode - opening
		# 	self.thinned_image = cv.bitwise_or(subset,self.thinned_image)
		# 	image_to_thin = erode.copy()
		# self.thinned_image = 255.0 - self.thinned_image

		self.thinned_image = np.where(skeletonize(self.enhanced_image/255), 0.0, 1.0)

		# Minutiae Extraction
		def cn(i, j, img):
			if img[i, j] == 0.0:
				offsets = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
						(0, 1),  (1, 1), (1, 0),        # p8    p4
                  		(1, -1), (0, -1), (-1, -1)] 	# p7 p6 p5
				pixel_values = [img[i+x, j+y] for x, y in offsets]
				sum_cn = 0.0
				for a in range(8):
					sum_cn += abs(pixel_values[a] - pixel_values[a+1])
				return sum_cn // 2
			return 2.0

		self.minutiae = {}
		self.minutiae_img = cv.cvtColor((255*self.thinned_image).astype(np.uint8), cv.COLOR_GRAY2RGB)
		for i in range(1, self.original_img.shape[0]-1):
			for j in range(1, self.original_img.shape[1]-1):
				current_cn = cn(i, j, self.thinned_image)
				if current_cn == 1 or current_cn == 3:
					self.minutiae[(i, j)] = current_cn

		# False minutiae removal
		# Close to boundary case
		minutiae_segment_mask = np.ones(self.original_img.shape)
		thinned_global_greyscale_variance = np.var(self.thinned_image)*0.1
		for i in range(0, self.original_img.shape[0], segment_block_size):
			for j in range(0, self.original_img.shape[1], segment_block_size):
				end_i = min(self.original_img.shape[0], i+segment_block_size)
				end_j = min(self.original_img.shape[1], j+segment_block_size)
				thinned_local_grayscale_variance = np.var(self.thinned_image[i: end_i, j: end_j])
				if thinned_local_grayscale_variance <= thinned_global_greyscale_variance:
					minutiae_segment_mask[i: end_i, j: end_j] = 0.0

		kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(2*segment_block_size, 2*segment_block_size))
		minutiae_segment_mask = cv.morphologyEx(minutiae_segment_mask, cv.MORPH_CLOSE, kernel_open_close)
		minutiae_segment_mask = cv.morphologyEx(minutiae_segment_mask, cv.MORPH_OPEN, kernel_open_close)

		# Outermost strip black
		for i in range(0, self.original_img.shape[0], segment_block_size):
			end_i = min(self.original_img.shape[0], i+segment_block_size)
			minutiae_segment_mask[i: end_i, 0: segment_block_size] = 0.0
			minutiae_segment_mask[i: end_i, self.original_img.shape[1]-segment_block_size: self.original_img.shape[1]] = 0.0

		for j in range(0, self.original_img.shape[1], segment_block_size):
			end_j = min(self.original_img.shape[1], i+segment_block_size)
			minutiae_segment_mask[0: segment_block_size, j: end_j] = 0.0
			minutiae_segment_mask[self.original_img.shape[0]-segment_block_size: self.original_img.shape[0], j:end_j] = 0.0

		new_minutiae = {}
		for (x, y) in self.minutiae:
			neighbourhood = [(0, 1), (0, -1), (0, 0), (1, 0), (-1, 0)]
			to_append = True
			for direction_x, direction_y in neighbourhood:
				try:
					if minutiae_segment_mask[x + direction_x*segment_block_size, y + direction_y*segment_block_size] == 0.0:
						to_append = False
						break
				except IndexError:
					to_append = False
					break
			if to_append:
				new_minutiae[(x, y)] = self.minutiae[(x, y)]
		self.minutiae = new_minutiae

		not_visited_pixels = np.ones(self.original_img.shape)
		default_thresh = segment_block_size
		def removal_traversal(x, y, img, thresh, last_minutiae_type):
			not_visited_pixels[x, y] = 0.0
			if thresh == 0:
				return 6.0
			neighbourhood = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1)]
			to_return_minutiae = 6.0
			for (d_x, d_y) in neighbourhood:
				n_x, n_y = x+d_x, y+d_y
				if img[n_x, n_y] == 0.0 and not_visited_pixels[x, y]:
					if (n_x, n_y) in self.minutiae:
						current_minutiae_type = self.minutiae[(n_x, n_y)]
						returned_minutiae = removal_traversal(n_x, n_y, img, default_thresh, current_minutiae_type)
						to_return_minutiae = min(current_minutiae_type, to_return_minutiae)
						if current_minutiae_type + last_minutiae_type <= 4.0 or current_minutiae_type + returned_minutiae <= 4.0:
							del self.minutiae[(n_x, n_y)]
					else:
						to_return_minutiae = min(removal_traversal(n_x, n_y, img, thresh-1, last_minutiae_type), to_return_minutiae)
			return to_return_minutiae

		# Close to eachother
		# Deleting while iterating
		for (x, y) in sorted(list(self.minutiae.keys()), key=lambda x: (x[0], x[1],)):
			if (x, y) in self.minutiae:
				if removal_traversal(x, y, self.thinned_image, default_thresh, self.minutiae[(x, y)]) <= 3.0:
					del self.minutiae[(x, y)]

		# Draw minutiae on image
		for (x, y) in self.minutiae:
			c_n = self.minutiae[(x, y)]
			if c_n == 1:
				cv.circle(self.minutiae_img, (y,x), radius=3, color=(0, 0, 255), thickness=1)
			if c_n == 3:
				cv.circle(self.minutiae_img, (y,x), radius=3, color=(0, 255, 0), thickness=1)

		cv.imshow("seg mask", self.segmentation_mask)
		cv.waitKey(0)
		cv.imshow("norm", self.normalized_image)
		cv.waitKey(0)
		cv.imshow("orient", self.orientation_map/np.pi)
		cv.waitKey(0)
		cv.imshow("freq map", self.frequency_map)
		cv.waitKey(0)
		cv.imshow("enhanced", self.enhanced_image)
		cv.waitKey(0)
		cv.imshow("thinned", self.thinned_image)
		cv.waitKey(0)
		cv.imshow("minutiae segment mask", minutiae_segment_mask)
		cv.waitKey(0)
		cv.imshow("minutiae", self.minutiae_img)
		cv.waitKey(0)
		# not_visited_pixels = cv.erode(not_visited_pixels, kernel_open_close)
		cv.imshow("not visited", not_visited_pixels)
		cv.waitKey(0)

	def alignment(self, other_fp):
		# Generalized Hough Transform
		# It is assumed both fingerprint have same size
		a = np.zeros((180, self.original_img.shape[0], self.original_img.shape[1]))
