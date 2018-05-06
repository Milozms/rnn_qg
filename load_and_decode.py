import tensorflow as tf
from utils import Dataset
from tqdm import tqdm
import nltk
import logging
import pickle
import numpy as np
import json
import os
from model import Model

def load(modelid):
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("./log/log.txt", mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)
	with open('./dicts/word2id.pickle', 'rb') as f:
		word2id = pickle.load(f)
	with open('./dicts/kb2id.pickle', 'rb') as f:
		kb2id = pickle.load(f)
	with open('./dicts/wordemb.pickle', 'rb') as f:
		wordemb = pickle.load(f)
	with open('./dicts/kbemb.pickle', 'rb') as f:
		kbemb = pickle.load(f)
	with open('./dicts/wordlist.json', 'r') as f:
		wordlist = json.load(f)
		wordlist = ['<EOS>', '<SOS>'] + wordlist
	flags = tf.flags
	flags.DEFINE_integer('hidden', 600, "")
	flags.DEFINE_integer('word_vocab_size', len(word2id), "")
	flags.DEFINE_integer('word_emb_dim', 300, "")
	flags.DEFINE_integer('kb_vocab_size', len(kb2id), "")
	flags.DEFINE_integer('kb_emb_dim', 100, "")
	flags.DEFINE_integer('maxlen', 35, "")
	flags.DEFINE_integer('batch', 128, "")
	flags.DEFINE_integer('epoch_num', 50, "")
	flags.DEFINE_boolean('is_train', False, "")
	flags.DEFINE_float('max_grad_norm', 0.1, "")
	flags.DEFINE_float('lr', 0.00025, "")
	config = flags.FLAGS
	valid_file = './sq/annotated_fb_data_train.txt'
	valid_dset = Dataset(valid_file, max_cnt=128)
	with tf.variable_scope('model'):
		model = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, './savemodel/model' + str(modelid) + '.pkl')
		out_idx = model.decode(sess, valid_dset)
		sentences = []
		for s in out_idx:
			words = []
			for w in s:
				words.append(wordlist[w])
			sentence = ' '.join(words)
			sentences.append(sentence)
		with open('output.json', 'w') as f:
			json.dump(sentences, f)

if __name__ == '__main__':
	load()
