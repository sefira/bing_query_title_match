import numpy as np
import re
import itertools
import logging
# for over-sampling imbalanced learning
from collections import Counter
import numpy as np
from sklearn.utils import check_random_state
from scipy.sparse import hstack,vstack

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def read_query_and_question(query_file, question_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    print(query_file)
    print(question_file)
    query_data = list(open(query_file, "r").readlines())
    query_data = [s.strip() for s in query_data]
    question_data = list(open(question_file, "r").readlines())
    question_data = [s.strip() for s in question_data]
    
    train_number = len(query_data)
    logger.info("Read {} lines query data, and {} question data".format(len(query_data), len(question_data)))

    if (len(query_data) != len(question_data)):
        logger.info("Can't read data, query data is inconsistent with question data")
        return None
    
    # filter too short data
    train_querys = []
    train_questions = []
    for i in range(len(query_data)):
        if (len(query_data[i]) <= 1) or (len(question_data[i]) <= 1):
            logger.info("Invalid train data in {}: {}, {}".format(i, query_data[i], question_data[i]))
            train_number = train_number - 1
            continue
        train_querys.append(query_data[i])
        train_questions.append(question_data[i])
    return [train_number, train_querys, train_questions]

def load_data(train_data_file, raw_data_file):
    N, querys, questions = read_query_and_question(train_data_file, raw_data_file)
    if (len(querys) != len(questions) or len(querys) != N or N != len(questions)):
        logger.info("Can't load data, number is inconsistent:{}:{}:{}".format(N, len(querys), len(questions)))
        return None

    # check if there is any wrong in the formated data and label
    random_state = check_random_state(12)
    check_index = random_state.randint(low=0, high=N-1,size=10)
    for i in check_index:
        logger.info("Print Query and Question for checking:{} <vs> {}".format(querys[i], questions[i]))

    return [N, querys, questions]

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
