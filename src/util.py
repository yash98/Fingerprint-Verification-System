# Unrelated helper functions
# Some general use function also taken from net
from math import sqrt
import cv2 as cv

def euclidean_distance(x1, y1, x2, y2):
	return sqrt((x2-x1)**2 + (y2-y1)**2)

def custom_round(x, base=5):
    return base * round(x/base)

def get_line_ends(i, j, W, tang):
	if -1 <= tang and tang <= 1:
		begin = (int((-W/2) * tang + j + W/2), i)
		end = (int((W/2) * tang + j + W/2), i+W)
	else:
		begin = (j + W//2, int(i + W/2 + W/(2 * tang)))
		end = (j - W//2, int(i + W/2 - W/(2 * tang)))
	return (begin, end)

def interactive_display(window_label, image):
	cv.imshow(window_label, image)
	while 1:
		key = cv.waitKey(0) & 0xFF
		# wait for ESC key to exit
		if key == 27:
			cv.destroyAllWindows()
			break
	cv.destroyAllWindows()
