import numpy as np
import pickle
import linecache
import re
from tqdm import tqdm

def ent_format_dot(ent):
	split_ent = ent.split('/')
	split_ent = split_ent[1:]
	return '.'.join(split_ent)

def pad_sequence(seq, maxlen):
	if len(seq)>maxlen:
		return seq[:maxlen]
	pad_len = maxlen - len(seq)
	return seq + [0]*pad_len

class Dataset(object):
	def __init__(self, filename, max_cnt = None, shuffle = True):
		filename += '_with_single_placeholder'
		with open('./dicts/word2id.pickle', 'rb') as f:
			word2id = pickle.load(f)
		with open('./dicts/kb2id.pickle', 'rb') as f:
			kb2id = pickle.load(f)
		triples = []
		questions = []
		maxlen = 0
		subnames = []
		for line in linecache.getlines(filename):
			line = line.strip()
			tokens = line.split('\t')
			subname = tokens[0]
			question = tokens[4]
			tokens = tokens[1:4]
			triple = []
			for tok in tokens:
				split_tok = tok.split('/')
				strip_tok = split_tok[1:]
				new_tok = '/' + '/'.join(strip_tok)
				try:
					kb_id = kb2id[new_tok]
					triple.append(kb_id)
				except:
					print('%s not exist' % new_tok)

			words_ = re.split('[^0-9a-zA-Z<>]+', question)
			words = []
			for word in words_:
				if word != '':
					word_lower = word.lower()
					try:
						word_id = word2id[word_lower]
						words.append(word_id)
					except:
						print('%s not exist' % word_lower)
			words_ = []
			triples.append(triple)
			questions.append(words)
			subnames.append(subname)
			if len(words) > maxlen:
				maxlen = len(words)
		self.data = []
		self.datasize = len(triples)
		if max_cnt != None and max_cnt < self.datasize:
			self.datasize = max_cnt
		for i in range(self.datasize):
			self.data.append((triples[i], pad_sequence(questions[i], 35), len(questions[i]) + 1, subnames[i]))
			# len + 1 because EOS
		self.maxlen = maxlen
		if shuffle:
			np.random.shuffle(self.data)
		self.current_index = 0

	def get_mini_batch(self, batch_size):
		if self.current_index >= self.datasize:
			return None
		if self.current_index + batch_size > self.datasize:
			batch = self.data[self.current_index:]
			self.current_index = self.datasize
		else:
			batch = self.data[self.current_index:self.current_index + batch_size]
			self.current_index += batch_size
		triples = []
		questions = []
		qlen = []
		subnames = []
		for ins in batch:
			triples.append(ins[0])
			questions.append(ins[1])
			qlen.append(ins[2])
			subnames.append(ins[3])
		return triples, questions, qlen, subnames


class testDataset(object):
	def __init__(self, filename, max_cnt = None, shuffle = True):
		fbsub2name = {}
		print('Reading freebase subject to name mapping......')
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

		with open('./dicts/word2id.pickle', 'rb') as f:
			word2id = pickle.load(f)
		with open('./dicts/kb2id.pickle', 'rb') as f:
			kb2id = pickle.load(f)
		triples = []
		questions = []
		maxlen = 0
		subnames = []
		for line in linecache.getlines(filename):
			line = line.strip()
			tokens = line.split('\t')
			question = tokens[3]
			tokens = tokens[0:3]
			triple = []
			for tok in tokens:
				# split_tok = tok.split('/')
				# strip_tok = split_tok[1:]
				# new_tok = '/' + '/'.join(strip_tok)
				try:
					kb_id = kb2id[tok]
					triple.append(kb_id)
				except:
					print('%s not exist' % tok)

			words_ = re.split('[^0-9a-zA-Z<>]+', question)
			words = []
			for word in words_:
				if word != '':
					words.append(word.lower())
			words_ = []
			try:
				subname = fbsub2name[ent_format_dot(tokens[0])]
			except:
				print('No name for entity %s' % ent_format_dot(tokens[0]))
				continue
			triples.append(triple)
			questions.append(words)
			subnames.append(subname)
			if len(words) > maxlen:
				maxlen = len(words)
		self.data = []
		self.datasize = len(triples)
		if max_cnt != None and max_cnt < self.datasize:
			self.datasize = max_cnt
		for i in range(self.datasize):
			self.data.append((triples[i], pad_sequence(questions[i], 35), len(questions[i]) + 1, subnames[i]))
			# len + 1 because EOS
		self.maxlen = maxlen
		if shuffle:
			np.random.shuffle(self.data)
		self.current_index = 0

	def get_mini_batch(self, batch_size):
		if self.current_index >= self.datasize:
			return None
		if self.current_index + batch_size > self.datasize:
			batch = self.data[self.current_index:]
			self.current_index = self.datasize
		else:
			batch = self.data[self.current_index:self.current_index + batch_size]
			self.current_index += batch_size
		triples = []
		questions = []
		qlen = []
		subnames = []
		for ins in batch:
			triples.append(ins[0])
			questions.append(ins[1])
			qlen.append(ins[2])
			subnames.append(ins[3])
		return triples, questions, qlen, subnames


