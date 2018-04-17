# coding: utf-8

# ---- Main for FQA ----
# -- Francis Fu Yao 
# -- francis_yao@pku.edu.cn
# -- MON 29TH AUG 2017

import tensorflow as tf 
from data_utils_full import Dataset
# from model import Model
from model_checklist import Model
import os
import cPickle
import time
GPU_NUM = 1
outrecord_no_kv = 1
outrecord_kv = 2
checklist_no_kv = 3
checklist_vanilla = 4
outrecord_kv_q_attn_mem = 5
outrecord_kv_out_attn_mem = 6
outrecord_no_kv_sigmoid = 7
no_outrecord = 8
attn_hist = 9
out_split = 10
out_split_simple = 11
out_split_full = 12
out_split_fullrecord = 13
out_split_fullrecord_implicitgate = 14
out_split_fullrecord_explicitgate = 15
outrecord_no_kv_out_attn_mem = 16
out_split_fullrecord_explicitgate_attn_mem = 17
model_configs = {1:  "outrecord_no_kv",
                 2:  "outrecord_kv",
                 3:  "checklist_no_kv",
                 4:  "checklist_vanilla",
                 5:  "outrecord_kv_q_attn_mem",
                 6:  "outrecord_kv_out_attn_mem",
                 7:  "outrecord_no_kv_sigmoid",
                 8:  "no_outrecord",
                 9:  "attn_hist",
                 10: "out_split", 
                 11: "out_split_simple",
                 12: "out_split_full",
                 13: "out_split_fullrecord",
                 14: "out_split_fullrecord_implicitgate", 
                 15: "out_split_fullrecord_explicitgate",
                 16: "outrecord_no_kv_out_attn_mem",
                 17: "out_split_fullrecord_explicitgate_attn_mem"}
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)
flags = tf.flags
flags.DEFINE_integer("vocab_size", 0, "")
flags.DEFINE_integer("vocab_norm_size", 0, "")
flags.DEFINE_integer("vocab_spec_size", 0, "")
flags.DEFINE_integer("max_q_len", 0, "")
flags.DEFINE_integer("max_a_len", 0, "")
flags.DEFINE_integer("max_s_len", 0, "")
flags.DEFINE_integer("state_size", 256, "")
flags.DEFINE_integer("epoch_num", 60, "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_boolean("is_train", True, "")
flags.DEFINE_boolean("is_kv", True, "")
flags.DEFINE_integer("qclass_size", 10, "")
flags.DEFINE_integer("aclass_size", 3, "")
flags.DEFINE_integer("memkey_size", 21, "")
flags.DEFINE_integer("attn_hop_size", 2, "")
flags.DEFINE_integer("out_mode_gate", False, "")
flags.DEFINE_integer("gpu", GPU_NUM, "")
flags.DEFINE_float("gpufrac", 0.45, "")
flags.DEFINE_integer("ncon_size", 10000, "")
flags.DEFINE_integer("conj_size", -1,"")
flags.DEFINE_integer("wiki_size", -1,"")
flags.DEFINE_integer("config_id", checklist_no_kv, "")
flags.DEFINE_boolean("out_kv", False, "")
flags.DEFINE_string("config_name", "", "")
flags.DEFINE_string("name_suffix", "wikidata", "")
flags.DEFINE_string("data_source", "ncon_conj", "")
# flags.DEFINE_string("data_source", "all", "")
flags.DEFINE_string("out_dir", "../output", "")
flags.DEFINE_string("model_dir", "../saved/", "")
flags.DEFINE_string("data_dir", "../data/dataset.pkl", "")
config = flags.FLAGS # question: in which function does these flags.DEFINE* take effect?

def main():
  # read dataset
  start_time = time.time()
  # dset = Dataset()
  dset = cPickle.load(open(config.data_dir, "rb"))
  dset.build_remain(config)
  print("\n%.2f seconds to read dset" % (time.time() - start_time))

  # build model
  config.config_name = model_configs[config.config_id] + "_" + config.name_suffix
  config.vocab_size = dset.total_words
  config.vocab_norm_size = dset.norm_word_cnt
  config.vocab_spec_size = dset.spec_word_cnt
  config.max_q_len = dset.max_q_len
  config.max_a_len = dset.max_a_len
  config.max_m_len = dset.max_mem_len
  with tf.variable_scope("model"):
    m = Model(config, "train")
    m.build()
  config.is_train = False
  with tf.variable_scope("model", reuse = True):
    mvalid = Model(config, "valid")
    print("building valid model")
    mvalid.build()
    # print("building test model")
    mtest = Model(config, "test")
    mtest.build()
  print("\ntime to build: %.2f\n\n" % (time.time() - start_time))
  # train
  m.train(dset, mvalid, mtest)
  return 

if __name__ == "__main__":
  main()

