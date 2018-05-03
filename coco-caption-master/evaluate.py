#coding=utf-8
#-*- coding: UTF-8 -*-
import copy 
import numpy as np
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import codecs

def getArgs():
	parse=argparse.ArgumentParser()
	parse.add_argument('--input', '-i', type=str, help='decode output', required=True)
	parse.add_argument('--golden', '-g', type=str, help='golden question', required=True)
	parse.add_argument('--output', '-o', type=str, help='file to store bleu_score', default='None')
	parse.add_argument('--version', '-v', type=str, help='different post processing', default='')
	args=parse.parse_args()
	return vars(args)

def main():
	args = getArgs()
	inFile = args['input']
	goldenFile = args['golden']
	outFile = args['output']
	if outFile == 'None':
		outFile = inFile + '-bleu' + args['version']
	annFile = inFile + '-annotations'
	resFile = inFile + '-results'
	print('inFile', inFile)
	print('outFile', outFile)
	decode = []
	with codecs.open(inFile, 'r', 'utf-8') as fh:
		for line in fh.readlines():
			decode.append(line.strip())
	golden = []
	with codecs.open(goldenFile, 'r', 'utf-8') as fg:
		for line in fg.readlines():
			golden.append(line.strip())

	ann = {}
	ann['info'] = {u'description': u'This is stable 1.0 version of the 2014 MS COCO dataset.', u'url': u'http://mscoco.org', u'version': u'1.0', u'year': 2014, u'contributor': u'Microsoft COCO group', u'date_created': u'2015-01-27 09:11:52.357475'}
	ann['type'] = u'captions'
	ann['licenses'] = [{u'url': u'http://creativecommons.org/licenses/by-nc-sa/2.0/', u'id': 1, u'name': u'Attribution-NonCommercial-ShareAlike License'}, {u'url': u'http://creativecommons.org/licenses/by-nc/2.0/', u'id': 2, u'name': u'Attribution-NonCommercial License'}, {u'url': u'http://creativecommons.org/licenses/by-nc-nd/2.0/', u'id': 3, u'name': u'Attribution-NonCommercial-NoDerivs License'}, {u'url': u'http://creativecommons.org/licenses/by/2.0/', u'id': 4, u'name': u'Attribution License'}, {u'url': u'http://creativecommons.org/licenses/by-sa/2.0/', u'id': 5, u'name': u'Attribution-ShareAlike License'}, {u'url': u'http://creativecommons.org/licenses/by-nd/2.0/', u'id': 6, u'name': u'Attribution-NoDerivs License'}, {u'url': u'http://flickr.com/commons/usage/', u'id': 7, u'name': u'No known copyright restrictions'}, {u'url': u'http://www.usa.gov/copyright.shtml', u'id': 8, u'name': u'United States Government Work'}]
	ann['images'] = []
	ann['annotations'] = []
	res = []
	image_example = {u'license': 3, u'url': u'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg', u'file_name': u'COCO_val2014_000000391895.jpg', u'height': 360, u'width': 640, u'date_captured': u'2013-11-14 11:18:45', u'id': 391895}
	annotation_example = {u'image_id': 203564, u'id': 37, u'caption': u'A bicycle replica with a clock as the front wheel.'}
	res_example = {u'image_id': 404464, u'caption': u'black and white photo of a man standing in front of a building'}
	
	for i in range(len(decode)):
		ti = copy.deepcopy(image_example)
		ti['id'] = i 
		ta = copy.deepcopy(annotation_example)
		ta['image_id'] = i 
		ta['image_id'] = i
		ta['caption'] = golden[i]
		tr = copy.deepcopy(res_example)
		tr['image_id'] = i 
		tr['caption'] = decode[i]
		ann['images'].append(ti)
		ann['annotations'].append(ta)
		res.append(tr)

	with codecs.open(annFile, 'w', 'utf-8') as fh:
		json.dump(ann, fh)
	with codecs.open(resFile, 'w', 'utf-8') as fh:
		json.dump(res, fh)

	coco = COCO(annFile)
	cocoRes = coco.loadRes(resFile)
	cocoEval = COCOEvalCap(coco, cocoRes)
	cocoEval.params['image_id'] = cocoRes.getImgIds()
	cocoEval.evaluate()

	fout = open(outFile, 'w') 
	# print output evaluation scores
	for metric, score in cocoEval.eval.items():
		fout.write('%s: %.3f\r\n'%(metric, score))
	fout.close()

if __name__ == '__main__':
	main()
