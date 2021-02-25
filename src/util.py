# Unrelated utility functions
from math import sqrt

def euclidean_distance(x1, y1, x2, y2):
	return sqrt((x2-x1)**2 + (y2-y1)**2)

def custom_round(x, base=5):
    return base * round(x/base)