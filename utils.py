import numpy as np
import pickle
import linecache
import re

def pad_sequence(seq, maxlen):
	if len(seq)>maxlen:
		return seq[:maxlen]
	pad_len = maxlen - len(seq)
	return seq + [0]*pad_len

class Dataset(object):
	def __init__(self, filename, max_cnt = None, shuffle = True):
		with open('./dicts/word2id.pickle', 'rb') as f:
			word2id = pickle.load(f)
		with open('./dicts/kb2id.pickle', 'rb') as f:
			kb2id = pickle.load(f)
		triples = []
		questions = []
		maxlen = 0
		for line in linecache.getlines(filename):
			line = line.strip()
			tokens = line.split('\t')
			question = tokens[3]
			tokens = tokens[:3]
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

			words_ = re.split('[^0-9a-zA-Z]+', question)
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
			if len(words) > maxlen:
				maxlen = len(words)
		self.data = []
		self.datasize = len(triples)
		if max_cnt != None and max_cnt < self.datasize:
			self.datasize = max_cnt
		for i in range(self.datasize):
			self.data.append((triples[i], pad_sequence(questions[i], 35), len(questions[i]) + 1))
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
		for ins in batch:
			triples.append(ins[0])
			questions.append(ins[1])
			qlen.append(ins[2])
		return triples, questions, qlen





