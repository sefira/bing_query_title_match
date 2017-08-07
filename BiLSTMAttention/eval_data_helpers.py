#! /usr/bin/env python

import logging
import os.path
import sys
import jieba
import re

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

def process_data(line):
    """
    word break and remove word
    Returns split sentences
    """
    query, question = line.split('\t')
    # Word break
    query_list = jieba.cut(query)
    question_list = jieba.cut(question)
    output_query = u' '.join(query_list) + '\n'
    output_question = u' '.join(question_list)
    # if(len(line) < 2):
    #     return "UNK"
    return output_query, output_question

def load_data(eval_data_file):
    eval_data = list(open(eval_data_file, "r").readlines())
    XY = [process_data(item) for item in eval_data]
    X = [item[0] for item in XY]
    Y = [item[1] for item in XY]
    return [len(XY), X, Y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
