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

def build_kb_list(
		files=('./sq/annotated_fb_data_test.txt',
			   './sq/annotated_fb_data_train.txt',
			   './sq/annotated_fb_data_valid.txt'),
		outfiles=('./dicts/entlist.json', './dicts/rellist.json')
):
	exist_entities = set()
	exist_relations = set()
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/entity2id.txt')):
		line = line.strip()
		tokens = line.split()
		exist_entities.add(tokens[0])
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/relation2id.txt')):
		line = line.strip()
		tokens = line.split()
		exist_relations.add(tokens[0])

	entset = set()
	relset = set()
	for infile in files:
		for line in tqdm(linecache.getlines(infile)):
			line = line.strip('\n')
			tokens = line.split('\t')
			tokens = tokens[0:3]
			triple = []
			for tok in tokens:
				# split_tok = tok.split('/')
				# strip_tok = split_tok[1:]
				# new_tok = '/' + '/'.join(strip_tok)
				triple.append(tok)
			for ent in [triple[0], triple[2]]:
				if ent not in exist_entities:
					print(ent)
				entset.add(ent)
			for rel in [triple[1]]:
				if rel not in exist_relations:
					print(rel)
				relset.add(rel)
	print(len(entset))
	print(len(relset))
	with open(outfiles[0], 'w') as f:
		json.dump(list(entset), f)
	with open(outfiles[1], 'w') as f:
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
	not_pretrained = []
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
			not_pretrained.append(w)

	print(pretrained, vocab_size)
	with open('./dicts/wordemb.pickle', 'wb') as f:
		pickle.dump(emb, f)
	with open('./dicts/word2id.pickle', 'wb') as f:
		pickle.dump(word2id, f)
	with open('./dicts/word2id.json', 'w') as f:
		json.dump(word2id, f)
	with open('./dicts/not_pretrained.json', 'w') as f:
		json.dump(not_pretrained, f)

def build_kb_dict_emb(
		listfiles = ('./dicts/entlist.json', './dicts/rellist.json'),
		idfile = './dicts/kb2id.pickle',
		embfile = './dicts/kbemb.pickle'
):
	with open(listfiles[0], 'r') as f:
		entlist = json.load(f)
	with open(listfiles[1], 'r') as f:
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
	ent_pretrained = 0
	emb = np.zeros([kb_size, dim])

	emb_whole = []
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/entity2vecfb5m.vec')):
		line = line.strip()
		tokens = line.split()
		emb_whole.append([float(x) for x in tokens])
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/entity2id.txt')):
		line = line.strip()
		tokens = line.split()
		if tokens[0] in kb2id:
			emb[kb2id[tokens[0]]] = np.array(emb_whole[int(tokens[1])])
			ent_pretrained += 1
			initialized[tokens[0]] = True
	print('%d entities pretrained, %d entities total' % (ent_pretrained, len(entlist)))

	rel_pretrained = 0
	emb_whole = []
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/relation2vecfb5m.vec')):
		line = line.strip()
		tokens = line.split()
		emb_whole.append([float(x) for x in tokens])
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/relation2id.txt')):
		line = line.strip()
		tokens = line.split()
		if tokens[0] in kb2id:
			emb[kb2id[tokens[0]]] = np.array(emb_whole[int(tokens[1])])
			rel_pretrained += 1
			initialized[tokens[0]] = True
	print('%d relations pretrained, %d relations total' % (rel_pretrained, len(rellist)))

	with open(embfile, 'wb') as f:
		pickle.dump(emb, f, protocol=4)
	with open(idfile, 'wb') as f:
		pickle.dump(kb2id, f, protocol=4)

def sq_triple_format(infile):
	# infile = 'train', 'test', 'valid'
	outf = open('./sq/sq'+ infile +'.txt', 'w')
	for line in tqdm(linecache.getlines('./sq/annotated_fb_data_'+infile+'.txt')):
		line = line.strip('\n')
		tokens = line.split('\t')
		outtokens = []
		for tok in tokens[0:3]:
			split_tok = tok.split('/')
			strip_tok = split_tok[1:]
			new_tok = '/' + '/'.join(strip_tok)
			outtokens.append(new_tok)
		outtokens.append(tokens[3])
		outf.write('\t'.join(outtokens) + '\n')
	outf.close()

def newdata_kb():
	build_kb_list(
		files=('./sq/sqtest.txt',
			   './sq/sqtrain.txt',
			   './sq/sqvalid.txt',
			   './sq/newdata.txt'),
		outfiles=('./dicts/entlist.json', './dicts/rellist.json'))
	build_kb_dict_emb(listfiles=('./dicts/entlist.json', './dicts/rellist.json'),
					  idfile='./dicts/kb2id.pickle',
					  embfile='./dicts/kbemb.pickle')

if __name__ == '__main__':
	newdata_kb()
	# build_word_list()
	# build_word_dict_emb()
	# sq_triple_format('train')
	# sq_triple_format('test')
	# sq_triple_format('valid')