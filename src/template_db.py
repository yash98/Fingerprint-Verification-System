# Class for management of template fingerprints
from . import fingerprint

import os
import re

class template_database:
	def __init__(self, directory_name):
		self.db = {}
		collection = [os.path.join(directory_name, file) for file in os.listdir(directory_name)]

		fingerprint_id_extractor = re.compile(r".*/<(?P<fp_id>[0-9]+)>_<(?P<instance_id>[0-9]+)>+\.tif")
		for template_file in collection:
			extracted_fp_id = fingerprint_id_extractor.findall(template_file)
			if len(extracted_fp_id) != 1:
				continue
			if extracted_fp_id[0][0] == '' or extracted_fp_id[0][1] == '':
				continue

			fp_id = fingerprint_id_extractor[0][0]
			if fp_id not in self.db:
				self.db[fp_id] = {}
			
			template = fingerprint.fp(template_file)
			self.db[fp_id][extracted_fp_id[0][1]] = template
	
	def match(self, query_fp):
		scores = {}
		for fp_id in self.db:
			instances = self.db[fp_id]
			scores[fp_id] = {}
			for instance_id in instances:
				scores[fp_id][instance_id] = query_fp.match(query_fp, instances[instance_id])
		
		return scores 
