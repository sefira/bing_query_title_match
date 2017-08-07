#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import eval_data_helpers
from word2vec_helpers import Word2VecHelper
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("eval_data_file",        "../data/eval_data.txt", "Data source for the eval")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size",           1,                      "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir",        "runs/20170807040037",  "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,                   "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,                  "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
eval_size, eval_query, eval_question = eval_data_helpers.load_data(FLAGS.eval_data_file)

max_document_length = 50
word2vec_helpers = Word2VecHelper()
x = word2vec_helpers.SentencesIndex(eval_query, max_document_length)
y = word2vec_helpers.SentencesIndex(eval_question, max_document_length)

# Checkpoint
ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, 'checkpoints'))
if ckpt:
    print("Read model parameters from %s" % ckpt.model_checkpoint_path)

# Evaluation
# ==================================================
print("\nEvaluating...\n")

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_xpos = graph.get_operation_by_name("input_xpos").outputs[0]
        input_xneg = graph.get_operation_by_name("input_xneg").outputs[0]
        real_len_x = graph.get_operation_by_name("real_len_x").outputs[0]
        real_len_xpos = graph.get_operation_by_name("real_len_xpos").outputs[0]
        real_len_xneg = graph.get_operation_by_name("real_len_xneg").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        batch_size = graph.get_operation_by_name("batch_size").outputs[0]

        def real_len_func(batches):
            return [np.argmin(batch + [0]) for batch in batches]

        # Tensors we want to evaluate
        x_vs_xpos = graph.get_operation_by_name("output/x_vs_xpos").outputs[0]
        x_vs_xneg = graph.get_operation_by_name("output/x_vs_xneg").outputs[0]
        attention_weight_x = graph.get_operation_by_name("attention/attention_x/attention_weights").outputs[0]
        attention_weight_xpos = graph.get_operation_by_name("attention/attention_xpos/attention_weights").outputs[0]
        attention_weight_xneg = graph.get_operation_by_name("attention/attention_xneg/attention_weights").outputs[0]
        
        # Collect the predictions here
        all_x_vs_xpos = []
        all_attention_weight_x = []
        all_attention_weight_xpos = []
        for i in range(len(x)):
            real_len_x_value = real_len_func([x[i]])
            real_len_y_value = real_len_func([y[i]])
            feed_dict = {
                input_x: [x[i]],
                input_xpos: [y[i]],
                real_len_x: real_len_x_value,
                real_len_xpos: real_len_y_value,
                dropout_keep_prob: 1.0,
                batch_size: 1,
            }
            batch_x_vs_xpos, batch_attention_weight_x, batch_attention_weight_xpos = \
            sess.run([x_vs_xpos, attention_weight_x, attention_weight_xpos], feed_dict)

            all_attention_weight_x.append(batch_attention_weight_x)
            all_attention_weight_xpos.append(batch_attention_weight_xpos)
            all_x_vs_xpos.append(batch_x_vs_xpos)
            print("=====================================================")
            print(x[i])
            print(eval_query[i])
            print(batch_attention_weight_x[0][0:real_len_x_value[0]])
            print(y[i])
            print(eval_question[i])
            print(batch_attention_weight_xpos[0][0:real_len_y_value[0]])
