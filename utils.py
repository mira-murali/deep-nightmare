import time
import os

def create_dir(name, parent=None):
	"""
	function to create directory at required depth
	if parent exists, ./parent/name is created else ./name is created
	You dont have to create parent
	TODO: Yuhan
	"""
	now=time.time()
	full_path=''
	if parent is not None and os.path.isdir('./'+parent):
		full_path = './'+parent+'/'+name
		os.mkdir(full_path)
		return full_path
	full_path = './'+name
	os.mkdir(full_path)
	return full_path