# a web server for simliarty matching in query and top10 BingAPI retrieval

import tensorflow as tf
import numpy as np 
import os 
import datetime
import eval_data_helpers
from word2vec_helpers import Word2VecHelper
import csv
import jieba
import re
from flask import Flask
from flask import request
import requests
import lxml.html
import urllib

# Parameter
# =================================================
# Data Parameter
tf.flags.DEFINE_string("eval_data_file",        "../data/eval_data.txt",    "Data source for the eval")
tf.flags.DEFINE_integer("batch_size",           10,                         "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir",        "runs/20170807040037",      "Chechpoint drectory from training runs")
tf.flags.DEFINE_integer("max_document_length",  50,                         "max document length")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,                       "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,                      "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

max_document_length = FLAGS.max_document_length
word2vec_helpers = Word2VecHelper()

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config = session_conf)
    with sess.as_default():

        # checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, 'checkpoints'))
        if ckpt:
            print("Read model parameters from {}".format(ckpt.model_checkpoint_path))

        saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_xpos = graph.get_operation_by_name("input_xpos").outputs[0]
        real_len_x = graph.get_operation_by_name("real_len_x").outputs[0]
        real_len_xpos = graph.get_operation_by_name("real_len_xpos").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        batch_size = graph.get_operation_by_name("batch_size").outputs[0]

        def real_len_func(batchs):
            return [np.argmin(batch + [0]) for batch in batchs]

        # tensor we want to evaluate
        x_vs_xpos = graph.get_operation_by_name("output/x_vs_xpos").outputs[0]

def eval():
    # load data
    eval_size, eval_query, eval_question = eval_data_helpers.load_data(FLAGS.eval_data_file)
    x = word2vec_helpers.SentencesIndex(eval_query, max_document_length)
    x = [x[8]] * FLAGS.batch_size
    y = word2vec_helpers.SentencesIndex(eval_question, max_document_length)
    y = y[0 : FLAGS.batch_size]

    # eval
    print("\nEvaluating...\n")

    real_len_x_value = real_len_func(x)
    real_len_xpos_value = real_len_func(y)
    feed_dict = {
        input_x: x,
        input_xpos: y,
        real_len_x: real_len_x_value,
        real_len_xpos: real_len_xpos_value,
        dropout_keep_prob: 1.0,
        batch_size: FLAGS.batch_size,
    }
    batch_x_vs_xpos = sess.run([x_vs_xpos], feed_dict)
    print(batch_x_vs_xpos)
    top1 = np.argmax(batch_x_vs_xpos)
    print(top1)

def process_line(line):
    # Word break
    line_list = jieba.cut(line)
    output_line = u' '.join(line_list)
    return output_line

def predict(query, questions):
    questions = questions[0 : FLAGS.batch_size]
    y = [process_line(item) for item in questions]
    y = word2vec_helpers.SentencesIndex(y, max_document_length)
    x = process_line(query)
    x = word2vec_helpers.SentencesIndex([x, x + " UNK"], max_document_length)
    x = [x[0]] * (FLAGS.batch_size)
    
    real_len_x_value = real_len_func(x)
    real_len_xpos_value = real_len_func(y)
    feed_dict = {
        input_x: x,
        input_xpos: y,
        real_len_x: real_len_x_value,
        real_len_xpos: real_len_xpos_value,
        dropout_keep_prob: 1.0,
        batch_size: FLAGS.batch_size,
    }
    batch_x_vs_xpos = sess.run([x_vs_xpos], feed_dict)
    print(batch_x_vs_xpos)
    top1 = np.argmax(batch_x_vs_xpos)
    print(top1)
    return batch_x_vs_xpos, top1

def get_query_retrieval(query):
    url = "https://www.bing.com/api/v6/search?q=" + query + "%20site:zhidao.baidu.com&appid=371E7B2AF0F9B84EC491D731DF90A55719C7D209&mkt=zh-cn&responsefilter=webpages"
    questions = []
    answers = []
    urls = []
    try:
        res = requests.get(url)
        print(res)
        res = res.json()["webPages"]["value"]
        if len(res) >= 10:
            for item in res:
                questions.append(item["name"].replace("百度知道", "").replace("全球最大中文互动问答平台", ""))
                answers.append(item["snippet"])
                urls.append(item["displayUrl"])
    except:
        pass

    return questions, answers, urls

def get_full_answer(url):
    print(url)
    page = lxml.html.document_fromstring(urllib.request.urlopen(url).read())
    best = page.xpath("//pre[contains(@class, 'best-text mb-10')]")
    common = page.xpath("//meta[contains(@name, 'description')]")
    if len(best) >= 1:
        best = best[0].text_content()
    else:
        if len(common) >= 1:
            best = common[0].text_content()
        else:
            best = "没有查询到答案"
    return best

web_server = Flask(__name__)

@web_server.route('/')
def hello_world():
    return 'Hello World!'

@web_server.route("/query")
def get_query():
    query = request.args.get('q')
    questions, answers, urls = get_query_retrieval(query)
    if len(questions) < 10:
        return "没有查询到答案"
    batch_similarity, top1 = predict(query, questions)
    reply = get_full_answer(urls[top1])
    return reply

if __name__ == "__main__":
    #eval()
    web_server.run(host='0.0.0.0', port=9006, debug=True)
