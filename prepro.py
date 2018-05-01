import json
import pickle
import linecache
from tqdm import tqdm
import re
import numpy as np

def build_word_list():
	wordset = set()
	files = ['./sq/annotated_fb_data_test.txt_with_single_placeholder',
			 './sq/annotated_fb_data_train.txt_with_single_placeholder',
			 './sq/annotated_fb_data_valid.txt_with_single_placeholder']
	for infile in files:
		for line in linecache.getlines(infile):
			line = line.strip('\n')
			tokens = line.split('\t')
			question = tokens[4]
			words_ = re.split('[^0-9a-zA-Z<>]+', question)
			words = []
			for word in words_:
				if word != '':
					words.append(word.lower())
			words_ = []
			for word in words:
				wordset.add(word)
	print(len(wordset))
	with open('./dicts/wordlist.json', 'w') as f:
		json.dump(list(wordset), f)

def build_kb_list():
	entset = set()
	relset = set()
	files = ['./sq/annotated_fb_data_test.txt',
			 './sq/annotated_fb_data_train.txt',
			 './sq/annotated_fb_data_valid.txt']
	for infile in files:
		for line in linecache.getlines(infile):
			line = line.strip('\n')
			tokens = line.split('\t')
			tokens = tokens[1:4]
			triple = []
			for tok in tokens:
				split_tok = tok.split('/')
				strip_tok = split_tok[1:]
				new_tok = '/' + '/'.join(strip_tok)
				triple.append(new_tok)
			for ent in [triple[0], triple[2]]:
				entset.add(ent)
			for rel in [triple[1]]:
				relset.add(rel)
	print(len(entset))
	print(len(relset))
	with open('./dicts/entlist.json', 'w') as f:
		json.dump(list(entset), f)
	with open('./dicts/rellist.json', 'w') as f:
		json.dump(list(relset), f)

def build_word_dict_emb():
	dim = 300
	with open('./dicts/wordlist.json', 'r') as f:
		wordlist = json.load(f)
	# word 0 is the end of sentence
	# word 1 is the start of sentence
	wordlist = ['<EOS>', '<SOS>'] + wordlist
	word2id = {}
	for i, word in enumerate(wordlist):
		word2id[word] = i
	vocab_size = len(wordlist)
	emb = np.zeros([vocab_size, dim])
	initialized = {}
	pretrained = 0
	avg_sigma = 0
	avg_mu = 0
	for line in tqdm(linecache.getlines('/Users/zms/Documents/学习资料/NLP/glove.840B.300d.txt')):
		line = line.strip()
		tokens = line.split()
		word = tokens[0]
		if word in word2id:
			vec = np.array([float(tok) for tok in tokens[-dim:]])
			wordid = word2id[word]
			emb[wordid] = vec
			initialized[word] = True
			pretrained += 1
			mu = vec.mean()
			sigma = np.std(vec)
			avg_mu += mu
			avg_sigma += sigma
	avg_sigma /= 1. * pretrained
	avg_mu /= 1. * pretrained
	for w in word2id:
		if w not in initialized:
			emb[word2id[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
	print(pretrained, vocab_size)
	with open('./dicts/wordemb.pickle', 'wb') as f:
		pickle.dump(emb, f)
	with open('./dicts/word2id.pickle', 'wb') as f:
		pickle.dump(word2id, f)
	with open('./dicts/word2id.json', 'w') as f:
		json.dump(word2id, f)

def build_kb_dict_emb():
	with open('./dicts/entlist.json', 'r') as f:
		entlist = json.load(f)
	with open('./dicts/rellist.json', 'r') as f:
		rellist = json.load(f)
	kb2id = {}
	dim = 100
	kb_size = len(entlist) + len(rellist)
	emb = np.zeros([kb_size, dim])
	for i, ent in enumerate(entlist):
		kb2id[ent] = i
	for i, rel in enumerate(rellist):
		kb2id[rel] = i + len(entlist)
	initialized = {}
	pretrained = 0
	emb = np.zeros([kb_size, dim])

	emb_whole = []
	for line in tqdm(linecache.getlines('/home/laiyx/data/TransE/FB5M/entity2vecfb5m.vec')):
		line = line.strip()
		tokens = line.split()
		emb_whole.append([float(x) for x in tokens])
	for line in tqdm(linecache.getlines('/home/laiyx/data/TransE/FB5M/entity2id.txt')):
		line = line.strip()
		tokens = line.split()
		if tokens[0] in kb2id:
			emb[kb2id[tokens[0]]] = np.array(emb_whole[int(tokens[1])])
			pretrained += 1
			initialized[tokens[0]] = True
	print(pretrained, len(entlist))

	emb_whole = []
	for line in tqdm(linecache.getlines('/home/laiyx/data/TransE/FB5M/relation2vecfb5m.vec')):
		line = line.strip()
		tokens = line.split()
		emb_whole.append([float(x) for x in tokens])
	for line in tqdm(linecache.getlines('/home/laiyx/data/TransE/FB5M/relation2id.txt')):
		line = line.strip()
		tokens = line.split()
		if tokens[0] in kb2id:
			emb[kb2id[tokens[0]]] = np.array(emb_whole[int(tokens[1])])
			pretrained += 1
			initialized[tokens[0]] = True
	print(pretrained, kb_size)

	with open('./dicts/kbemb.pickle', 'wb') as f:
		pickle.dump(emb, f)
	with open('./dicts/kb2id.pickle', 'wb') as f:
		pickle.dump(kb2id, f)

if __name__ == '__main__':
	build_kb_list()
	build_kb_dict_emb()
