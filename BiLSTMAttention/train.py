#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import logging
import data_helpers
from word2vec_helpers import Word2VecHelper
from BiLSTMAttention import BiLSTMAttention
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile

# Parameters
# ==================================================

# Data loading params

flags = tf.flags
FLAGS = flags.FLAGS

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage",  0.8,          "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("query_file",            "../data/2017-07-27-22-01_Query.tsv.wordbreak", "Data source for the train data.")
tf.flags.DEFINE_string("question_file",         "../data/2017-07-27-22-01_Question.tsv.wordbreak", "Data source for the label data.")
tf.flags.DEFINE_string("toy_query_file",        "../data/2017-07-27-22-01_Query.tsv.toy.wordbreak", "Toy Data source for the train data.")
tf.flags.DEFINE_string("toy_question_file",     "../data/2017-07-27-22-01_Question.tsv.toy.wordbreak", "Toy Data source for the label data.")

# Model Hyperparameters
flags.DEFINE_float('lr',                    1e-2,       "The learning reate")
flags.DEFINE_integer("embedding_dim",       300,        "Dimensionality of character embedding (default: 128)")
flags.DEFINE_integer('hidden_layer_size',   300 ,       "LSTM hidden layer size")
flags.DEFINE_integer('attention_size',      600 ,       "Attention model size, double hidden size due to Bidirect")
flags.DEFINE_float('dropout_keep_prob',     0.5,        "Dropout rate")
flags.DEFINE_float('l2_reg_lambda',         0.0,        "l2 reg lambda")
flags.DEFINE_bool('non_static',             True,       "Whether change word2vec")
flags.DEFINE_bool('GRU',                    True,       "Whether use GRU")

# Training parameters
flags.DEFINE_integer("batch_size",          256,        "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs",          1,          "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every",      10000000,   "Evaluate model after X steps (default: 100)")
flags.DEFINE_integer("checkpoint_every",    5000,       "Save model after X steps (default: 100)")
flags.DEFINE_integer("num_checkpoints",     5,          "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_string("checkpoint",            '',         "Resume checkpoint")
tf.flags.DEFINE_boolean("allow_soft_placement", True,       "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,      "Log placement of ops on devices")

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))
logger.info("")


# Data Preparation
# ==================================================

# Load data
logger.info("Loading data...")
train_size, train_query, train_question = data_helpers.load_data(
    FLAGS.query_file, FLAGS.question_file)

# Build vocabulary
max_query_length = max([len(x.split()) for x in train_query])
max_question_length = max([len(x.split()) for x in train_question])
max_document_length = max(max_query_length, max_question_length)
word2vec_helpers = Word2VecHelper()
x = word2vec_helpers.SentencesIndex(train_query, max_document_length)
y = word2vec_helpers.SentencesIndex(train_question, max_document_length)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(x)))
query = x[shuffle_indices]
question_pos = y[shuffle_indices]
shuffle_indices = np.random.permutation(np.arange(len(x)))
question_neg = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(query)))
x_train, x_dev = query[:dev_sample_index], query[dev_sample_index:]
pos_train, pos_dev = question_pos[:dev_sample_index], question_pos[dev_sample_index:]
neg_train, neg_dev = question_neg[:dev_sample_index], question_neg[dev_sample_index:]
logger.info("Vocabulary Size: {:d}".format(word2vec_helpers.vocab_size))
logger.info("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        rnn = BiLSTMAttention(
            embedding_mat=word2vec_helpers.wordvector.astype(np.float32),
            non_static=FLAGS.non_static,
            GRU=FLAGS.GRU,
            sequence_length=max_document_length,
            hidden_layer_size=FLAGS.hidden_layer_size,
            vocab_size=word2vec_helpers.vocab_size,
            embedding_size=FLAGS.embedding_dim,
            attention_size=FLAGS.attention_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        if FLAGS.checkpoint == "":
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            logger.info("Writing to {}\n".format(out_dir))
        else:
            out_dir = FLAGS.checkpoint

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint, 'checkpoints'))
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        def real_len(batches):
            return [np.argmin(batch + [0]) for batch in batches]
        
        def train_step(x_batch, pos_batch, neg_batch):
            """
            A single training step
            """
            feed_dict = {
              rnn.input_x: x_batch,
              rnn.input_xpos: pos_batch,
              rnn.input_xneg: neg_batch,
              rnn.real_len_x: real_len(x_batch),
              rnn.real_len_xpos: real_len(pos_batch),
              rnn.real_len_xneg: real_len(neg_batch),
              rnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              rnn.batch_size: len(x_batch),
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_dev, pos_dev, neg_dev):
            """
            Evaluates model on a dev set
            """
            batches = data_helpers.batch_iter(
                list(zip(x_dev, pos_dev, neg_dev)), FLAGS.batch_size, 1)
            loss_sum = 0
            accuracy_sum = 0
            count = 0
            for batch in batches:
                x_batch, pos_batch, neg_batch = zip(*batch)
                feed_dict = {
                  rnn.input_x: x_batch,
                  rnn.input_xpos: pos_batch,
                  rnn.input_xneg: neg_batch,
                  rnn.real_len_x: real_len(x_batch),
                  rnn.real_len_xpos: real_len(pos_batch),
                  rnn.real_len_xneg: real_len(neg_batch),
                  rnn.dropout_keep_prob: 1.0,
                  rnn.batch_size: len(x_batch),
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict)
                loss_sum = loss_sum + loss
                accuracy_sum = accuracy_sum + loss
                count = count + 1
            loss = loss_sum / count
            accuracy = accuracy_sum / count
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, pos_train, neg_train)), FLAGS.batch_size, FLAGS.num_epochs)
        logger.info("With {} batch size, and {} train samples".format(FLAGS.batch_size, len(x_train)))
        logger.info("We get {} batches per epoch".format(len(x_train)/FLAGS.batch_size))

        # Training loop. For each batch...
        for batch in batches:
            x_batch, pos_batch, neg_batch = zip(*batch)
            train_step(x_batch, pos_batch, neg_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                logger.info("\nEvaluation:")
                dev_step(x_dev, pos_dev, neg_dev)
                logger.info("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))

        logger.info("Final Checkpoint:")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        logger.info("Saved model checkpoint to {}\n".format(path))
