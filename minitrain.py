import tensorflow as tf
from utils import Dataset
from tqdm import tqdm
import nltk
import logging
import pickle
import numpy as np
import json
import os
import math
from model import Model

def minitrain(config, train_file, valid_file, wordlist, kblist):
	train_dset = Dataset(train_file)
	valid_dset = Dataset(valid_file, shuffle=False)
	with tf.variable_scope('model'):
		model = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)
	config.is_train = False
	with tf.variable_scope('model', reuse=True):
		mtest = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)

	saver = tf.train.Saver()
	tfconfig = tf.ConfigProto()
	# tfconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=tfconfig)
	# writer = tf.summary.FileWriter('./graph', sess.graph)
	sess.run(tf.global_variables_initializer())
	num_batch = int(math.ceil(train_dset.datasize / model.batch))
	for ei in range(model.epoch_num):
		train_dset.current_index = 0
		loss_iter = 0.0
		for bi in tqdm(range(num_batch)):
			mini_batch = train_dset.get_mini_batch(model.batch)
			if mini_batch == None:
				break
			triples, questions, qlen, subnames = mini_batch
			feed_dict = {}
			feed_dict[model.triple] = triples
			feed_dict[model.question] = questions
			feed_dict[model.qlen] = qlen
			feed_dict[model.keep_prob] = 1.0
			loss, train_op, out_idx = sess.run(model.out, feed_dict=feed_dict)
			# writer.add_graph(sess.graph)
			loss_iter += loss
		loss_iter /= num_batch
		logging.info('iter %d, train loss: %f' % (ei, loss_iter))
		model.valid_model(sess, valid_dset, ei, saver)
		if ei % 5 == 0:
			mtest.decode_test_model(sess, valid_dset, ei, wordlist, kblist, saver)

if __name__ == '__main__':
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
	with open('./dicts/entlist.json', 'r') as f:
		entlist = json.load(f)
	with open('./dicts/rellist.json', 'r') as f:
		rellist = json.load(f)
	kblist = entlist + rellist
	flags = tf.flags
	flags.DEFINE_integer('hidden', 600, "")
	flags.DEFINE_integer('word_vocab_size', len(word2id), "")
	flags.DEFINE_integer('word_emb_dim', 300, "")
	flags.DEFINE_integer('kb_vocab_size', len(kb2id), "")
	flags.DEFINE_integer('kb_emb_dim', 100, "")
	flags.DEFINE_integer('maxlen', 35, "")
	flags.DEFINE_integer('batch', 100, "")
	flags.DEFINE_integer('epoch_num', 30, "")
	flags.DEFINE_boolean('is_train', True, "")
	flags.DEFINE_float('max_grad_norm', 0.1, "")
	flags.DEFINE_float('lr', 0.00025, "")
	config = flags.FLAGS
	train_file = './sq/annotated_fb_data_train.txt'
	valid_file = './sq/annotated_fb_data_valid.txt'
	minitrain(config, train_file, valid_file, wordlist, kblist)
