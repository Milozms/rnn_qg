import tensorflow as tf
from utils import Dataset
from tqdm import tqdm
import nltk
import logging
import pickle
import numpy as np
import json
import os

def bleu_val(ques, out_idx):
	ques = list(ques)
	out_idx = list(out_idx)
	while ques[-1] == 0:
		ques.pop()
	while out_idx[-1] == 0:
		out_idx.pop()
	sf = nltk.translate.bleu_score.SmoothingFunction()
	return nltk.translate.bleu(references=[ques], hypothesis=out_idx, smoothing_function=sf.method1)

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
								   
	def build(self):
		self.triple = tf.placeholder(dtype = tf.int32, shape = [None, 3], name = 'triple')
		self.question = tf.placeholder(dtype = tf.int32, shape = [None, self.maxlen], name = 'question')
		self.qlen = tf.placeholder(dtype = tf.int32, shape = [None], name = 'question_len')
		self.keep_prob = tf.placeholder(dtype = tf.float32, shape = ())
		batch_size = tf.shape(self.triple)[0]

		hidden = self.hidden
		word_vocab_size = self.word_vocab_size
		word_emb_dim = self.word_emb_dim
		kb_vocab_size = self.kb_vocab_size
		kb_emb_dim = self.kb_emb_dim
		maxlen = self.maxlen

		with tf.device("/cpu:0"):
			with tf.variable_scope("embeddings"):
				word_embeddings = tf.get_variable(name = "word_embedding",
									dtype = tf.float32,
									initializer = tf.constant(self.word_emb_mat, dtype=tf.float32),
									trainable=False)
				kb_embeddings = tf.get_variable(name = "kb_embedding",
									dtype = tf.float32, 
									initializer = tf.constant(self.kb_emb_mat, dtype=tf.float32),
									trainable=False)
		
		trip_emb = tf.nn.embedding_lookup(kb_embeddings, self.triple) # [batch, 3, dim]
		ques_emb = tf.nn.embedding_lookup(word_embeddings, self.question)

		with tf.variable_scope("encoder"):
			#fact_embedding
			fact_emb = tf.reshape(trip_emb, [-1, kb_emb_dim*3], name="fact_embedding")
			fact = tf.layers.dense(fact_emb, hidden,
								   kernel_initializer = tf.random_normal_initializer())

		# attention
		def attention(query, step_i):
			with tf.variable_scope("attention") as att_scope:
				if step_i != 0:
					att_scope.reuse_variables()
				att_sim_w = tf.get_variable('att_sim_w', shape=[kb_emb_dim, hidden],
											initializer = tf.random_normal_initializer())
				att_sim_w = tf.tile(tf.expand_dims(att_sim_w, axis=0), [batch_size, 1, 1]) # [batch, dim, hidden]
				trip_mult_w = tf.matmul(trip_emb, att_sim_w) # [batch, 3, hidden]
				#trip_mult_w = tf.layers.dense(trip_emb, hidden, use_bias=False) # [batch, 3, hidden]
				query = tf.expand_dims(query, axis=2)  # [batch, hidden, 1]
				trip_mult_w_mult_query = tf.matmul(trip_mult_w, query) # [batch, 3, 1]
				trip_mult_w_mult_query = tf.reshape(trip_mult_w_mult_query, [-1, 3]) # [batch, 3]
				actived = tf.tanh(trip_mult_w_mult_query)
				# attention weight
				logits = tf.nn.softmax(actived, dim=1) # [batch, 3]
				att_w = tf.expand_dims(logits, axis=1) # [batch, 1, 3]
				# trip_emb [batch, 3, dim]
				att_o = tf.matmul(att_w, trip_emb) # [batch, 1, dim]
				att_o = tf.squeeze(att_o, axis=1) # [batch, dim]
			return att_o

		decoder_cell = tf.nn.rnn_cell.GRUCell(num_units = hidden)
		decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=self.keep_prob)
		dec_ques = tf.unstack(ques_emb, axis=1)
		prev_out = None
		out_idx = []
		loss_steps = []
		# pred_size = word_vocab_size + 1  #??????
		label_steps = tf.one_hot(self.question, word_vocab_size)
		label_steps = tf.unstack(label_steps, axis=1)
		# initial state
		prev_hidden = fact
		sos = tf.ones(shape=[batch_size], dtype=tf.int32)
		sos_emb = tf.nn.embedding_lookup(word_embeddings, sos)
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
				att_o = attention(prev_hidden, time_step)
				# concat attention (prev hidden) and current input
				cell_in = tf.concat([cur_in, att_o], 1)
				cur_out, cur_hidden = decoder_cell(cell_in, prev_hidden) # [batch, hidden]
				prev_hidden = cur_hidden
				# prev_out = cur_out

				# output projection to normal words
				output_w = tf.get_variable('output_w', shape=[hidden, word_vocab_size],
										   initializer = tf.random_normal_initializer())
				output_b = tf.get_variable('output_b', shape=[word_vocab_size],
										   initializer=tf.random_normal_initializer())
				output = tf.matmul(cur_out, output_w) + output_b # [batch, pred_size]
				output = tf.nn.softmax(output, dim=1)
				outputs.append(output)
				prev_out = tf.matmul(output, word_embeddings) # [batch, word_emb_dim]

				if self.is_train:
					labels = label_steps[time_step]
					loss_steps.append(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = output))

				out_index = tf.argmax(output, 1)
				out_idx.append(out_index)

		out_idx = tf.transpose(tf.stack(out_idx)) # [batch_size, timesteps]

		if self.is_train == False:
			self.out = out_idx
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


	def decode_test_model(self, sess, test_dset, niter, saver):
		test_dset.current_index = 0
		num_batch = int(test_dset.datasize / self.batch) + 1
		out_idx = []
		bleu = 0.0
		for bi in tqdm(range(num_batch)):
			triples, questions, qlen = test_dset.get_mini_batch(self.batch)
			feed_dict = {}
			feed_dict[self.triple] = triples
			feed_dict[self.question] = questions
			feed_dict[self.qlen] = qlen
			feed_dict[self.keep_prob] = 1.0
			loss, out_idx_cur = sess.run(self.out_valid, feed_dict=feed_dict)
			out_idx_cur = np.array(out_idx_cur, dtype=np.int32)
			out_idx_lst = [list(x) for x in out_idx_cur]
			out_idx += out_idx_lst
			for i in range(len(questions)):
				bleu += bleu_val(questions[i], out_idx_cur[i])
		bleu /= test_dset.datasize
		logging.info('iter %d, bleu = %f' % (niter, bleu))
		if bleu > self.maxbleu:
			self.maxbleu = bleu
			saver.save(sess, './savemodel/model0.pkl')
			with open('./output/out_idx.json', 'wb') as f:
				pickle.dump(out_idx, f)

	def valid_model(self, sess, valid_dset, niter, saver):
		valid_dset.current_index = 0
		num_batch = int(valid_dset.datasize / self.batch) + 1
		out_idx = []
		loss_iter = 0.0
		for bi in tqdm(range(num_batch)):
			triples, questions, qlen = valid_dset.get_mini_batch(self.batch)
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
		num_batch = int(train_dset.datasize / self.batch) + 1
		for ei in range(self.epoch_num):
			dset.current_index = 0
			loss_iter = 0.0
			for bi in tqdm(range(num_batch)):
				triples, questions, qlen = dset.get_mini_batch(self.batch)
				feed_dict = {}
				feed_dict[self.triple] = triples
				feed_dict[self.question] = questions
				feed_dict[self.qlen] = qlen
				feed_dict[self.keep_prob] = 0.9
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
	flags.DEFINE_integer('epoch_num', 50, "")
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



					







		