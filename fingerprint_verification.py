# Main file, user interaction channeled thorugh here
import src.fingerprint
import src.template_db

import argparse
import cv2 as cv

def showimage(img_label, img):
	cv.imshow(img_label, img)
	while True:
		k = cv.waitKey(0) & 0xFF
		if k == 27:         # wait for ESC key to exit
			cv.destroyAllWindows()
			break
	cv.destroyAllWindows()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Fingerprint Verification System.')
	parser.add_argument("template_fingerprints_directory", help="Directory containing template fingerprint image. tif extension file with <fingerprint_id>_<instance_id> as file name")
	parser.add_argument("query_fingerprint_image", help="This fingerprint impressions image is compared with the template database to verify identity")

	args = parser.parse_args()

	query_fp = src.fingerprint.fp(args.query_fingerprint_image)
	templates = src.template_db.template_database(args.template_fingerprints_directory)

	print(len(query_fp.minutiae))
	print(templates.match(query_fp))
