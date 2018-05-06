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
import copy
from tensorflow.contrib import crf

def bleu_val(ques, out_idx, bleu_order):
	sf = nltk.translate.bleu_score.SmoothingFunction()
	if bleu_order == 1:
		weight = (1, 0, 0, 0)
	elif bleu_order == 2:
		weight = (0.5, 0.5, 0, 0)
	elif bleu_order == 3:
		weight = (1.0/3, 1.0/3, 1.0/3, 0)
	elif bleu_order == 4:
		weight = (0.25, 0.25, 0.25, 0.25)
	else:
		weight = (0.25, 0.25, 0.25, 0.25)
	return nltk.translate.bleu(references=[ques], hypothesis=out_idx, smoothing_function=sf.method1, weights=weight)

def restore_placeholder(idx_seq, wordlist, subname):
	words = []
	for w in idx_seq:
		if w == 0:
			break
		if wordlist[w] == '<placeholder>':
			words += subname.split(' ')
		else:
			words.append(wordlist[w])
	return words

class Model(object):
	def __init__(self, config, kb_emb_mat, word_emb_mat):
		self.hidden = config.hidden
		self.word_vocab_size = config.word_vocab_size
		self.word_emb_dim = config.word_emb_dim
		self.kb_vocab_size = config.kb_vocab_size
		self.kb_emb_dim = config.kb_emb_dim
		self.batch = config.batch
		self.is_train = config.is_train
		self.maxlen = config.maxlen
		self.word_emb_mat = word_emb_mat
		self.kb_emb_mat = kb_emb_mat
		self.epoch_num = config.epoch_num
		self.max_grad_norm = config.max_grad_norm
		self.lr = config.lr
		self.maxbleu = 0.0
		self.minloss = 100
		self.build()

	def attention(self, query, step_i):
		batch_size = tf.shape(self.triple)[0]
		with tf.variable_scope("attention") as att_scope:
			if step_i != 0:
				att_scope.reuse_variables()
			att_sim_w = tf.get_variable('att_sim_w', shape=[self.kb_emb_dim, self.hidden],
										initializer=tf.random_normal_initializer())
			att_sim_w = tf.tile(tf.expand_dims(att_sim_w, axis=0), [batch_size, 1, 1])  # [batch, dim, hidden]
			trip_mult_w = tf.matmul(self.trip_emb, att_sim_w)  # [batch, 3, hidden]
			# trip_mult_w = tf.layers.dense(trip_emb, hidden, use_bias=False) # [batch, 3, hidden]
			query = tf.expand_dims(query, axis=2)  # [batch, hidden, 1]
			trip_mult_w_mult_query = tf.matmul(trip_mult_w, query)  # [batch, 3, 1]
			trip_mult_w_mult_query = tf.reshape(trip_mult_w_mult_query, [-1, 3])  # [batch, 3]
			actived = tf.tanh(trip_mult_w_mult_query)
			# attention weight
			logits = tf.nn.softmax(actived, dim=1)  # [batch, 3]
			att_w = tf.expand_dims(logits, axis=1)  # [batch, 1, 3]
			# trip_emb [batch, 3, dim]
			att_o = tf.matmul(att_w, self.trip_emb)  # [batch, 1, dim]
			att_o = tf.squeeze(att_o, axis=1)  # [batch, dim]

		# att_o = tf.zeros(shape=[batch_size, kb_emb_dim], dtype=tf.float32)
		return att_o


	def build(self):
		self.triple = tf.placeholder(dtype = tf.int32, shape = [None, 3], name = 'triple')
		self.question = tf.placeholder(dtype = tf.int32, shape = [None, self.maxlen], name = 'question')
		self.qlen = tf.placeholder(dtype = tf.int32, shape = [None], name = 'question_len')
		self.keep_prob = tf.placeholder(dtype = tf.float32, shape = ())
		batch_size = tf.shape(self.triple)[0]
		# batch_size = self.triple.shape[0].value

		hidden = self.hidden
		word_vocab_size = self.word_vocab_size
		word_emb_dim = self.word_emb_dim
		kb_vocab_size = self.kb_vocab_size
		kb_emb_dim = self.kb_emb_dim
		maxlen = self.maxlen

		with tf.device("/cpu:0"):
			with tf.variable_scope("embeddings"):
				self.word_embeddings = tf.get_variable(name = "word_embedding",
									dtype = tf.float32,
									initializer = tf.constant(self.word_emb_mat, dtype=tf.float32),
									trainable=False)
				kb_embeddings = tf.get_variable(name = "kb_embedding",
									dtype = tf.float32,
									initializer = tf.constant(self.kb_emb_mat, dtype=tf.float32),
									trainable=False)
		
		self.trip_emb = tf.nn.embedding_lookup(kb_embeddings, self.triple) # [batch, 3, dim]
		ques_emb = tf.nn.embedding_lookup(self.word_embeddings, self.question)

		with tf.variable_scope("encoder"):
			# fact_embedding
			fact_emb = tf.reshape(self.trip_emb, [-1, kb_emb_dim*3], name="fact_embedding")
			self.fact = tf.layers.dense(fact_emb, hidden,
								   kernel_initializer = tf.random_normal_initializer())

		# attention
		def attention_(query, step_i):
			with tf.variable_scope("attention") as att_scope:

				if step_i != 0:
					att_scope.reuse_variables()
				att_sim_w = tf.get_variable('att_sim_w', shape=[kb_emb_dim, hidden],
											initializer = tf.random_normal_initializer())
				att_sim_w = tf.tile(tf.expand_dims(att_sim_w, axis=0), [batch_size, 1, 1]) # [batch, dim, hidden]
				trip_mult_w = tf.matmul(self.trip_emb, att_sim_w) # [batch, 3, hidden]
				# trip_mult_w = tf.layers.dense(trip_emb, hidden, use_bias=False) # [batch, 3, hidden]
				query = tf.expand_dims(query, axis=2)  # [batch, hidden, 1]
				trip_mult_w_mult_query = tf.matmul(trip_mult_w, query) # [batch, 3, 1]
				trip_mult_w_mult_query = tf.reshape(trip_mult_w_mult_query, [-1, 3]) # [batch, 3]
				actived = tf.tanh(trip_mult_w_mult_query)
				# attention weight
				logits = tf.nn.softmax(actived, dim=1) # [batch, 3]
				att_w = tf.expand_dims(logits, axis=1) # [batch, 1, 3]
				# trip_emb [batch, 3, dim]
				att_o = tf.matmul(att_w, self.trip_emb) # [batch, 1, dim]
				att_o = tf.squeeze(att_o, axis=1) # [batch, dim]

				# att_o = tf.zeros(shape=[batch_size, kb_emb_dim], dtype=tf.float32)
			return att_o

		self.decoder_cell = tf.nn.rnn_cell.GRUCell(num_units = hidden)
		# self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=self.keep_prob)
		dec_ques = tf.unstack(ques_emb, axis=1)
		prev_out = None
		out_idx = []
		loss_steps = []
		# pred_size = word_vocab_size + 1  #??????
		# label_steps = tf.one_hot(self.question, word_vocab_size)
		label_steps = tf.unstack(self.question, axis=1)
		# initial state
		prev_hidden = self.fact
		# prev_hidden = decoder_cell.zero_state(batch_size, dtype=tf.float32)
		sos = tf.ones(shape=[batch_size], dtype=tf.int32)
		sos_emb = tf.nn.embedding_lookup(self.word_embeddings, sos)
		with tf.variable_scope("decoder") as decoder_scope:
			outputs = []
			for time_step in range(maxlen):
				if time_step >= 1:
					decoder_scope.reuse_variables()
				if time_step == 0:
					cur_in = sos_emb # <SOS>
				else:
					if self.is_train:
						cur_in = dec_ques[time_step - 1] # [batch, word_dim]
					else:
						cur_in = prev_out # [batch, word_dim]
				# attention
				att_o = attention_(prev_hidden, time_step)
				# concat attention (prev hidden) and current input
				cell_in = tf.concat([cur_in, att_o], 1)
				# cell_in = cur_in
				cur_out, cur_hidden = self.decoder_cell(cell_in, prev_hidden) # [batch, hidden]
				prev_hidden = cur_hidden

				# output projection to normal words
				output_w = tf.get_variable('output_w', shape=[hidden, word_vocab_size],
										   initializer = tf.random_normal_initializer())
				output_b = tf.get_variable('output_b', shape=[word_vocab_size],
										   initializer=tf.random_normal_initializer())
				output = tf.matmul(cur_out, output_w) + output_b # [batch, pred_size]
				output_softmax = tf.nn.softmax(output, dim=1)
				outputs.append(output_softmax)

				if self.is_train:
					labels = label_steps[time_step]
					loss_steps.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = output))

				out_index = tf.argmax(output, 1) # [batch, vocab_size]
				out_idx.append(out_index)
				# input for next cell
				# prev_out = tf.matmul(output_softmax, word_embeddings)  # [batch, word_emb_dim]
				prev_out = tf.nn.embedding_lookup(self.word_embeddings, out_index) # [batch, word_emb_dim]

		out_idx = tf.transpose(tf.stack(out_idx)) # [batch_size, timesteps]

		self.out_test = out_idx
		if self.is_train == False:
			return

		loss = tf.transpose(tf.stack(loss_steps)) # [batch_size, maxlen - 1]
		# mask loss
		loss_mask = tf.sequence_mask(self.qlen, maxlen, tf.float32)
		loss = tf.reduce_mean(loss_mask * loss)

		# self.out_test = [loss, out_idx]

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.max_grad_norm)
		train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
		self.out = [loss, train_op, out_idx]
		self.out_valid = [loss, out_idx]
		return

	def beam_search(self, sess, beam_size = 5):
		batch_size = tf.shape(self.triple)[0]
		hidden = self.hidden
		word_vocab_size = self.word_vocab_size
		word_emb_dim = self.word_emb_dim
		kb_vocab_size = self.kb_vocab_size
		kb_emb_dim = self.kb_emb_dim
		maxlen = self.maxlen

		with tf.variable_scope("decoder") as beam_scope:
			beam_scope.reuse_variables()

			sos = tf.ones(shape=[batch_size], dtype=tf.int32)
			sos_emb = tf.nn.embedding_lookup(self.word_embeddings, sos)

			prev_hidden = self.fact
			# time step 0: input <SOS>
			att_o = self.attention(prev_hidden, 0)
			cell_in = tf.concat([sos_emb, att_o], axis=1)
			cur_out, cur_hidden = self.decoder_cell(cell_in, prev_hidden)

			start_status = {
				'score': 0.0,
				'sequence': [],
				'prev_hidden': self.fact
			}
			beam_agenda = [start_status]

			while beam_agenda:
				next_agenda = []
				for status in beam_agenda:

					prev_hidden = status['prev_hidden']
					out_index = status['sequence'][-1] # [batch, vocab_size]
					time_step = len(status['sequence'])
					prev_out = tf.nn.embedding_lookup(self.word_embeddings, out_index)
					att_o = self.attention(prev_hidden, time_step)
					cell_in = tf.concat([prev_out, att_o], axis=1)
					cur_out, cur_hidden = self.decoder_cell(cell_in, prev_hidden)  # [batch, hidden]

					# output projection to normal words
					output_w = tf.get_variable('output_w', shape=[hidden, word_vocab_size],
											   initializer=tf.random_normal_initializer())
					output_b = tf.get_variable('output_b', shape=[word_vocab_size],
											   initializer=tf.random_normal_initializer())
					output = tf.matmul(cur_out, output_w) + output_b  # [batch, pred_size]
					output_softmax = tf.nn.softmax(output, dim=1)


	def decode_test_model(self, sess, test_dset, niter, wordlist, kblist, saver, dir = './output'):
		'''
		greedy search
		'''
		test_dset.current_index = 0
		num_batch = int(math.ceil(test_dset.datasize / self.batch))
		out_idx = []
		triples_idx = []
		bleu1 = 0.0
		bleu2 = 0.0
		bleu3 = 0.0
		bleu4 = 0.0
		outf = open(dir + '/output' + str(niter) + '.txt', 'w')
		for bi in tqdm(range(num_batch)):
			mini_batch = test_dset.get_mini_batch(self.batch)
			if mini_batch == None:
				break
			triples, questions, qlen, subnames = mini_batch
			feed_dict = {}
			feed_dict[self.triple] = triples
			# feed_dict[self.question] = questions
			feed_dict[self.qlen] = qlen
			feed_dict[self.keep_prob] = 1.0
			out_idx_cur = sess.run(self.out_test, feed_dict=feed_dict)
			out_idx_cur = np.array(out_idx_cur, dtype=np.int32)
			out_idx_lst = [list(x) for x in out_idx_cur]
			out_idx += out_idx_lst
			triples_idx += triples
			for i in range(len(questions)):
				question_restore = restore_placeholder(questions[i], wordlist, subnames[i])
				output_restore = restore_placeholder(out_idx_cur[i], wordlist, subnames[i])
				output_question = ' '.join(output_restore)
				outf.write(output_question + '\n')
				# bleu1 += bleu_val(question_restore, output_restore, 1)
				# bleu2 += bleu_val(question_restore, output_restore, 2)
				# bleu3 += bleu_val(question_restore, output_restore, 3)
				bleu4 += bleu_val(question_restore, output_restore, 4)
		# bleu1 /= test_dset.datasize
		# bleu2 /= test_dset.datasize
		# bleu3 /= test_dset.datasize
		bleu4 /= test_dset.datasize
		logging.info('iter %d, bleu4 = %f' % (niter, bleu4))
		if bleu4 > self.maxbleu:
			self.maxbleu = bleu4
			saver.save(sess, './savemodel/model' + str(niter) + '.pkl')
		outf.close()


	def valid_model(self, sess, valid_dset, niter, saver):
		valid_dset.current_index = 0
		num_batch = int(math.ceil(valid_dset.datasize / self.batch))
		out_idx = []
		loss_iter = 0.0
		for bi in tqdm(range(num_batch)):
			mini_batch = valid_dset.get_mini_batch(self.batch)
			if mini_batch == None:
				break
			triples, questions, qlen, subnames = mini_batch
			feed_dict = {}
			feed_dict[self.triple] = triples
			feed_dict[self.question] = questions
			feed_dict[self.qlen] = qlen
			feed_dict[self.keep_prob] = 1.0
			loss, out_idx_cur = sess.run(self.out_valid, feed_dict=feed_dict)
			loss_iter += loss
		loss_iter /= num_batch
		logging.info('iter %d, valid loss = %f' % (niter, loss_iter))
		if loss_iter < self.minloss:
			self.minloss = loss_iter
			saver.save(sess, './savemodel/model'+str(niter)+'.pkl')


	def train(self, dset, valid_dset):
		saver = tf.train.Saver()
		tfconfig = tf.ConfigProto()
		# tfconfig.gpu_options.allow_growth = True
		sess = tf.Session(config=tfconfig)
		sess.run(tf.global_variables_initializer())
		num_batch = int(dset.datasize / self.batch) + 1
		for ei in range(self.epoch_num):
			dset.current_index = 0
			loss_iter = 0.0
			for bi in tqdm(range(num_batch)):
				mini_batch = train_dset.get_mini_batch(self.batch)
				if mini_batch == None:
					break
				triples, questions, qlen, subnames = mini_batch
				feed_dict = {}
				feed_dict[self.triple] = triples
				feed_dict[self.question] = questions
				feed_dict[self.qlen] = qlen
				feed_dict[self.keep_prob] = 1.0
				loss, train_op, out_idx = sess.run(self.out, feed_dict=feed_dict)
				loss_iter += loss
			loss_iter /= num_batch
			logging.info('iter %d, train loss: %f' % (ei, loss_iter))
			self.valid_model(sess, valid_dset, ei, saver)
			# mtest.test_model(sess, test_dset, ei, saver)


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
	flags = tf.flags
	flags.DEFINE_integer('hidden', 600, "")
	flags.DEFINE_integer('word_vocab_size', len(word2id), "")
	flags.DEFINE_integer('word_emb_dim', 300, "")
	flags.DEFINE_integer('kb_vocab_size', len(kb2id), "")
	flags.DEFINE_integer('kb_emb_dim', 100, "")
	flags.DEFINE_integer('maxlen', 35, "")
	flags.DEFINE_integer('batch', 128, "")
	flags.DEFINE_integer('epoch_num', 200, "")
	flags.DEFINE_boolean('is_train', True, "")
	flags.DEFINE_float('max_grad_norm', 0.1, "")
	flags.DEFINE_float('lr', 0.00025, "")
	config = flags.FLAGS
	train_file = './sq/annotated_fb_data_train.txt'
	valid_file = './sq/annotated_fb_data_valid.txt'
	train_dset = Dataset(train_file)
	valid_dset = Dataset(valid_file)
	with tf.variable_scope('model'):
		model = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)
	# config.is_train = False
	# with tf.variable_scope('model', reuse=True):
	# 	mtest = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)
	model.train(train_dset, valid_dset)



