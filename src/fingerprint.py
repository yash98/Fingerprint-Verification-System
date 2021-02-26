# File for Base class to store fingerprint
from . import util

import sys
import math
import numpy as np
import scipy
from scipy.ndimage.interpolation import rotate
from skimage.morphology import skeletonize
import cv2 as cv
import fingerprint_enhancer
np.set_printoptions(threshold=sys.maxsize)

def get_line_ends(i, j, W, tang):
	if -1 <= tang and tang <= 1:
		begin = (int((-W/2) * tang + j + W/2), i)
		end = (int((W/2) * tang + j + W/2), i+W)
	else:
		begin = (j + W//2, int(i + W/2 + W/(2 * tang)))
		end = (j - W//2, int(i + W/2 - W/(2 * tang)))
	return (begin, end)

class fp:
	def __init__(self, path_fp_img, segment_block_size=15):
		self.original_img = cv.imread(path_fp_img, cv.IMREAD_GRAYSCALE)
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
		required_variance = 5000.0
		required_mean = 127.0
		current_variance = np.var(self.segmented_image)
		current_mean = np.mean(self.segmented_image)
		temp_dev = ((required_variance/current_variance)*((self.segmented_image-current_mean)**2)) ** 0.5
		self.normalized_image = np.where(self.segmented_image > current_mean, required_mean + temp_dev, required_mean - temp_dev)

		# Ridge orientation calculation
		grad_x = cv.Sobel(self.normalized_image/255, cv.CV_64F, 0, 1, ksize=3)
		grad_y = cv.Sobel(self.normalized_image/255, cv.CV_64F, 1, 0, ksize=3)

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

		self.orientation_image = cv.cvtColor((self.normalized_image).astype(np.uint8), cv.COLOR_GRAY2RGB)
		# self.orientation_image = np.ones(self.normalized_image.shape)
		for i in range(0, self.original_img.shape[0], segment_block_size):
			for j in range(0, self.original_img.shape[1], segment_block_size):
				end_i = min(self.original_img.shape[0], i+segment_block_size)
				end_j = min(self.original_img.shape[1], j+segment_block_size)
				line_direction = np.average(self.orientation_map[i:end_i, j:end_j])
				begin, end = get_line_ends(i, j, self.segment_block_size, math.tan(line_direction))
				cv.line(self.orientation_image, begin, end, (255, 0, 0), 1)

		# Frequency map calculation
		self.frequency_map = np.zeros(self.original_img.shape)
		min_wavelength, max_wavelength = 3, 10
		for i in range(0, self.original_img.shape[0], segment_block_size):
			for j in range(0, self.original_img.shape[1], segment_block_size):
				end_i = min(self.original_img.shape[0], i+segment_block_size)
				end_j = min(self.original_img.shape[1], j+segment_block_size)
				segment_block = self.normalized_image[i:end_i, j:end_j]
				orientation = np.mean(self.orientation_map[i: end_i, j:end_j])
				rotated_block = scipy.ndimage.rotate(segment_block, orientation*180/np.pi + 90, axes=(1, 0), reshape = False, order = 3, mode = 'nearest')
				# # Crop
				# cropsize = round(self.orientation_image.shape[1]/np.sqrt(2))
				# offset = round((self.orientation_image.shape[1]-cropsize)/2)
				# rotated_block = rotated_block[offset:offset+cropsize][offset:offset+cropsize]
				#  Peak calculation
				ridge_sum = np.sum(rotated_block, axis = 0)
				dilation = scipy.ndimage.grey_dilation(ridge_sum, 5, structure=np.ones(5))
				ridge_noise = np.abs(dilation - ridge_sum)
				peak_thresh = 2
				max_points = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
				max_index = np.where(max_points)
				_, num_peaks = np.shape(max_index)
				# Set frequency in map according to map
				if num_peaks < 2:
					self.frequency_map[i: end_i, j: end_j] = 0.0
				else:
					wavelength = (max_index[0][-1] - max_index[0][0])/(num_peaks - 1)
					if min_wavelength <= wavelength <= max_wavelength:
						self.frequency_map[i: end_i, j: end_j] = 1.0 / wavelength
					else:
						self.frequency_map[i: end_i, j: end_j] = 0.0
				
		# Apply Gabor filter
		# self.enhanced_image = np.zeros(self.original_img.shape)
		# for i in range(0, self.original_img.shape[0], segment_block_size):
		# 	for j in range(0, self.original_img.shape[1], segment_block_size):
		# 		end_i = min(self.original_img.shape[0], i+segment_block_size)
		# 		end_j = min(self.original_img.shape[1], j+segment_block_size)
		# 		average_orientation = np.mean(self.orientation_map[i:end_i, j:end_j])
		# 		gabor_kernel = cv.getGaborKernel((segment_block_size, segment_block_size), 5*self.frequency_map[i, j], average_orientation, 10.0*self.frequency_map[i, j], 1.0)
		# 		# gabor_kernel = cv.getGaborKernel((segment_block_size, segment_block_size), 5.0, average_orientation, 1.0, 1.0)
		# 		self.enhanced_image[i: end_i, j: end_j] = cv.filter2D(self.normalized_image[i: end_i, j: end_j], cv.CV_64F, gabor_kernel)
		# self.gabor_filter()
		
		self.enhanced_image = fingerprint_enhancer.enhance_Fingerprint(self.normalized_image)
		# cv.imshow("enhanced", self.enhanced_image)
		# cv.waitKey(0)

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
					self.minutiae[(i, j)] = (current_cn, self.orientation_map[i, j])

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

		# Removal Outermost strip black
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

		# Removal Image graph traversal
		# not_visited_pixels = np.ones(self.original_img.shape)
		# default_thresh = segment_block_size
		# def removal_traversal(x, y, img, thresh, last_minutiae_type):
		# 	not_visited_pixels[x, y] = 0.0
		# 	if thresh == 0:
		# 		return 6.0
		# 	neighbourhood = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1)]
		# 	to_return_minutiae = 6.0
		# 	for (d_x, d_y) in neighbourhood:
		# 		n_x, n_y = x+d_x, y+d_y
		# 		if img[n_x, n_y] == 0.0 and not_visited_pixels[x, y]:
		# 			if (n_x, n_y) in self.minutiae:
		# 				current_minutiae_type, _ = self.minutiae[(n_x, n_y)]
		# 				returned_minutiae = removal_traversal(n_x, n_y, img, default_thresh, current_minutiae_type)
		# 				to_return_minutiae = min(current_minutiae_type, to_return_minutiae)
		# 				if current_minutiae_type + last_minutiae_type <= 4.0 or current_minutiae_type + returned_minutiae <= 4.0:
		# 					del self.minutiae[(n_x, n_y)]
		# 			else:
		# 				to_return_minutiae = min(removal_traversal(n_x, n_y, img, thresh-1, last_minutiae_type), to_return_minutiae)
		# 	return to_return_minutiae

		# Close to eachother
		# Deleting while iterating
		# for (x, y) in sorted(list(self.minutiae.keys()), key=lambda x: (x[0], x[1],)):
		# 	if (x, y) in self.minutiae:
		# 		if removal_traversal(x, y, self.thinned_image, default_thresh, self.minutiae[(x, y)][0]) <= 3.0:
		# 			del self.minutiae[(x, y)]

		# Removal Cluster Centroid
		def cluster_removal():
			minutiae_list = list(self.minutiae.items())
			dist_thresh = self.segment_block_size/4
			cluster_found = False
			cluster_list = set()
			# centroid_sum = None
			# centroid = None
			for i in range(1, len(minutiae_list)):
				for j in range(0, i):
					(x1, y1), (_, _) = minutiae_list[i]
					(x2, y2), (_, _) = minutiae_list[j]
					dist = util.euclidean_distance(x1, y1, x2, y2)
					if dist <= dist_thresh:
						cluster_found = True
						cluster_list.add((x1, y1))
						cluster_list.add((x2, y2))
			
			if not cluster_found:
				return False

			for _ in range(10):
				for i in range(len(minutiae_list)):
					if (x1, y1) not in cluster_list:
						for (x2, y2) in cluster_list:
							(x1, y1), _ = minutiae_list[i]
							dist = util.euclidean_distance(x1, y1, x2, y2)
							if dist <= dist_thresh:
								cluster_list.add((x1, y1))

			for (x1, y1) in cluster_list:
				del self.minutiae[(x1, y1)]

			return True

		cluster_remain = True
		while cluster_remain:
			cluster_remain = cluster_removal()

		# Draw minutiae on image
		for (x, y) in self.minutiae:
			c_n, _ = self.minutiae[(x, y)]
			if c_n == 1:
				cv.circle(self.minutiae_img, (y,x), radius=3, color=(0, 0, 255), thickness=1)
			if c_n == 3:
				cv.circle(self.minutiae_img, (y,x), radius=3, color=(0, 255, 0), thickness=1)

		# cv.imshow("seg mask", self.segmentation_mask)
		# cv.waitKey(0)
		# cv.imshow("norm", self.normalized_image)
		# cv.waitKey(0)
		# cv.imshow("orient", self.orientation_map/np.pi)
		# cv.waitKey(0)
		# cv.imshow("or", self.orientation_image)
		# cv.waitKey(0)
		# cv.imshow("freq map", self.frequency_map)
		# cv.waitKey(0)
		# cv.imshow("enhanced", self.enhanced_image)
		# cv.waitKey(0)
		# cv.imshow("thinned", self.thinned_image)
		# cv.waitKey(0)
		# cv.imshow("minutiae segment mask", minutiae_segment_mask)
		# cv.waitKey(0)
		# cv.imshow("minutiae", self.minutiae_img)
		# cv.waitKey(0)
		# # not_visited_pixels = cv.erode(not_visited_pixels, kernel_open_close)(
	
	def alignment(self, other_fp):
		# Generalized Hough Transform
		# It is assumed both fingerprint have same size
		# query_fp.alignment(template_fp)
		accumulator = {}

		for (xt, yt), (_, theta_t) in other_fp.minutiae.items():
			for (xq, yq), (_, theta_q) in self.minutiae.items():
				d_theta = abs(theta_t - theta_q)
				d_theta = min(d_theta, 2*np.pi - d_theta)
				d_x = xt - xq*math.cos(d_theta) + yq*math.sin(d_theta)
				d_y = yt - xq*math.sin(d_theta) - yq*math.cos(d_theta)
				conf = util.custom_round(180*d_theta/np.pi, 2), util.custom_round(d_x, self.segment_block_size), util.custom_round(d_y, self.segment_block_size//4)
				if conf in accumulator:
					accumulator[conf] += 1
				else:
					accumulator[conf] = 1
		
		(theta, x, y) = max(accumulator, key=accumulator.get)
		return np.pi*theta/180, x, y
	
	def pair(self, other_fp, transform_config):
		flag_q = np.zeros((len(self.minutiae),))
		flag_t = np.zeros(len(other_fp.minutiae),)
		count_matched = 0
		matched_minutiae = []

		angle_thresh = 20 * np.pi / 180
		distance_thresh = self.segment_block_size/2

		ht_theta, ht_x, ht_y = transform_config
		i = 0
		for (xt, yt), (_, theta_t) in other_fp.minutiae.items():
			j = 0
			for (xq, yq), (_, theta_q) in self.minutiae.items():
				d_theta = abs(abs(theta_t - theta_q) - ht_theta)
				d_theta = min(d_theta, 2*np.pi - d_theta)
				d_x = xt - xq*math.cos(ht_theta) + yq*math.sin(ht_theta) - ht_x
				d_y = yt - xq*math.sin(ht_theta) - yq*math.cos(ht_theta) - ht_y

				if flag_t[i] == 0.0 and flag_q[j] == 0.0 and util.euclidean_distance(0, 0, d_x, d_y) <= distance_thresh and abs(d_theta) <= abs(angle_thresh):
					flag_t[i] = 1.0
					flag_t[i] = 1.0
					count_matched += 1
					matched_minutiae.append(((xt, yt), (xq, yq)))
				j += 1
			i += 1
		
		return count_matched, i

	def match(self, other_fp):
		return self.pair(other_fp, self.alignment(other_fp))