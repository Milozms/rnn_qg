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

def load():
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, './savemodel/model46.pkl')
