import json
import pickle
import linecache
from tqdm import tqdm
import re
import numpy as np
from urllib.request import urlopen

def query_entity_name(ent):
	try:
		url = 'http://www.wikidata.org/wiki/Special:EntityData/' + ent
		ent_json_str = urlopen(url).read()
		ent_json = json.loads(ent_json_str)
		name = ent_json['entities'][ent]['labels']['en']['value'].lower()
	except:
		name = None
		print('%s not found' % ent)
	return name

def sq_subjects_names_extractor():
	subject2name = {}
	files = ['./sq/annotated_fb_data_test.txt', './sq/annotated_fb_data_train.txt', './sq/annotated_fb_data_valid.txt']
	for infile in files:
		for line in linecache.getlines(infile):
			line = line.strip('\n')
			tokens = line.split('\t')
			sub = tokens[0]
			split_sub = sub.split('/')
			sub_conc = '.'.join(split_sub[1:])
			subject2name[sub_conc] = None

	namecnt = 0
	for line in linecache.getlines('./dicts/fb_en_title_final.txt'):
		line = line.strip()
		tokens = line.split()
		if tokens[0] in subject2name:
			subject2name[tokens[0]] = tokens[1]
			namecnt += 1

	print('number of subjects: %d' % len(subject2name))
	print('number of names: %d' % namecnt)
	# with open('sub2name.json', 'w') as f:
	# 	json.dump(subject2name, f)
	with open('notfound.txt', 'w') as f:
		files = ['./sq/annotated_fb_data_test.txt', './sq/annotated_fb_data_train.txt',
				 './sq/annotated_fb_data_valid.txt']
		for infile in files:
			for line in linecache.getlines(infile):
				strip_line = line.strip('\n')
				tokens = strip_line.split('\t')
				sub = tokens[0]
				split_sub = sub.split('/')
				sub_conc = '.'.join(split_sub[1:])
				if subject2name[sub_conc] == None:
					f.write(line)

def sq_subjects_names_extractor_from_wikidata():
	files = ['annotated_fb_data_test.txt',
			 'annotated_fb_data_train.txt',
			 'annotated_fb_data_valid.txt']
	for infile in files:
		with open('./sq/'+infile+'.with_name', 'w') as fout:
			wikidata_lines = linecache.getlines('./sq_wikidata/'+infile)
			for idx, line in enumerate(linecache.getlines('./sq/'+infile)):
				wikidata_line = wikidata_lines[idx]
				wikidata_line = wikidata_line.strip('\n')
				tokens = wikidata_line.split('\t')
				sub = tokens[0]
				sub_name = query_entity_name(sub)
				fout.write(sub_name+'\t'+line)

def read_fb2w():
	file = './fb2w/fb2w.nt'
	fb2w = {}
	for idx, line in enumerate(tqdm(linecache.getlines(file))):
		if idx <= 3:
			continue
		line = line.strip()
		tokens = line.split()
		fb = tokens[0][1:-1].split('/')[-1]
		wk = tokens[2][1:-1].split('/')[-1]
		fb2w[fb] = wk
	with open('./fb2w/fb2w.json','w') as f:
		json.dump(fb2w, f)
	return fb2w

def sq_subjects_to_wikidata():
	fbsub2name = {}
	for line in tqdm(linecache.getlines('./dicts/fb_en_title_final.txt')):
		line = line.strip()
		tokens = line.split('\t')
		if len(tokens) > 1:
			name = re.sub('\(.*\)', '', tokens[1]).strip()
			fbsub2name[tokens[0]] = name
	for line in tqdm(linecache.getlines('./dicts/entity_2.txt')):
		line = line.strip()
		tokens = line.split('\t')
		if len(tokens) > 1:
			name = tokens[1]
			fbsub2name[tokens[0]] = name


	files = ['./sq/annotated_fb_data_test.txt', './sq/annotated_fb_data_train.txt', './sq/annotated_fb_data_valid.txt']
	ncnt = 0
	addcnt = 0
	for infile in files:
		outfile = open(infile + '_with_subject_name', 'w')
		for idx, line in enumerate(tqdm(linecache.getlines(infile))):
			tokens = line.strip('\n').split('\t')
			sub = tokens[0]
			split_sub = sub.split('/')
			sub_conc = '.'.join(split_sub[1:])
			if sub_conc in fbsub2name:
				outfile.write(fbsub2name[sub_conc] + '\t' + line)
				# outfile.write(str(idx) + '\t' + fbsub2name[sub_conc] + '\t' + line)
			# elif sub_conc in fb2w:
			# 	addcnt += 1
			# 	wk = fb2w[sub_conc]
			# 	name = query_entity_name(wk)
			# 	if name != None:
			# 		outfile.write(name+'\t'+line)
			# 	else:
			# 		outfile.write('<WK>'+wk + '\t' + line)
			else:
				ncnt += 1
	print('not found number: %d' % ncnt)
	print('add number: %d' % addcnt)

def add_single_placeholder():
	files = ['./sq/annotated_fb_data_test.txt', './sq/annotated_fb_data_train.txt', './sq/annotated_fb_data_valid.txt']
	ncnt = 0
	for infile in files:
		file_withname = infile + '_with_subject_name'
		outfile = open(infile + '_with_single_placeholder', 'w')
		for idx, line in enumerate(tqdm(linecache.getlines(file_withname))):
			tokens = line.strip('\n').split('\t')
			subname = tokens[0].lower()
			question = tokens[4].lower()
			sub_idx = question.find(subname)
			if sub_idx == -1:
				words_ = re.split('[^0-9a-zA-Z<>]+', question)
				words = []
				for word in words_:
					if word != '':
						words.append(word.lower())
				words_ = []
				question = ' '.join(words)
				words_ = re.split('[^0-9a-zA-Z<>]+', subname)
				words = []
				for word in words_:
					if word != '':
						words.append(word.lower())
				words_ = []
				subname = ' '.join(words)
				# in case that subname is part of other word !!!
				sub_idx = question.find(subname)
				if sub_idx == -1:
					print('Subject not found: %s, %s' % (tokens[4], tokens[0]))
					ncnt += 1
				else:
					question = question.replace(subname, ' <PLACEHOLDER> ')
					outfile.write('\t'.join(tokens[:4])+'\t'+question+'\n')
			else:
				question = question.replace(subname, ' <PLACEHOLDER> ')
				outfile.write('\t'.join(tokens[:4]) + '\t' + question + '\n')
	print('no subject number: %d' % ncnt)


if __name__ == '__main__':
	# fb2w = read_fb2w()
	sq_subjects_to_wikidata()
	add_single_placeholder()
