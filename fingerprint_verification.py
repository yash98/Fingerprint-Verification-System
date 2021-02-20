# Main file, user interaction channeled thorugh here
import src.fingerprint

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Fingerprint Verification System.')
	parser.add_argument("target_fingerprint_image", help="This fingerprint impressions image is compared with the template database to verify identity")

	args = parser.parse_args()

	target_fp = src.fingerprint.fp(args.target_fingerprint_image, 10)

