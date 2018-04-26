import json
import pickle
import linecache
from tqdm import tqdm
import re
import numpy as np

def mp_category():
	type_set = set()
	files = ['./sq/annotated_fb_data_test.txt', './sq/annotated_fb_data_train.txt', './sq/annotated_fb_data_valid.txt']
	for infile in files:
		for line in linecache.getlines(infile):
			line = line.strip('\n')
			tokens = line.split('\t')
			pred = tokens[1]
			split_pred = pred.split('/')
			if split_pred[1] in ['cvg', 'base', 'user']:
				type = split_pred[2]
			else:
				continue
			# 	type = split_pred[1]
			# type = split_pred[1]
			type_set.add(type)
	print(type_set)
	print((len(type_set)))

if __name__ == '__main__':
	mp_category()