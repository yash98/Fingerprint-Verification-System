# Main file, user interaction channeled thorugh here
import src.fingerprint
import src.template_db
import src.util

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Fingerprint Verification System. Checks how many minutiae match between fingerprint 1 and fingerprint 2')
	# parser.add_argument("template_fingerprints_directory", help="Directory containing template fingerprint image. tif extension file with <fingerprint_id>_<instance_id> as file name")
	# parser.add_argument("query_fingerprint_image", help="This fingerprint impressions image is compared with the template database to verify identity")
	parser.add_argument("fingerprint_1", help="Path to fingerprint 1")
	parser.add_argument("fingerprint_2", help="Path to fingerprint 2")

	args = parser.parse_args()

	# query_fp = src.fingerprint.fp(args.query_fingerprint_image)
	# templates = src.template_db.template_database(args.template_fingerprints_directory)

	# print(len(query_fp.minutiae))
	# print(templates.match(query_fp))

	fp1 = src.fingerprint.fp(args.fingerprint_1)
	fp2 = src.fingerprint.fp(args.fingerprint_2)
	matches, _ = fp1.match(fp2)

	print("Fingerprint 1 has {} minutiae points".format(len(fp1.minutiae)))
	print("Fingerprint 2 has {} minutiae points".format(len(fp2.minutiae)))
	print("{} out of those match in between them".format(matches))
	print("{:.3f} percentage of minutiae matched".format(100*matches/min(len(fp1.minutiae), len(fp2.minutiae))))
	src.util.interactive_display("Fingerprint 1 Minutiae", fp1.minutiae_img)
	src.util.interactive_display("Fingerprint 2 Minutiae", fp2.minutiae_img)
