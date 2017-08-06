import logging
import os.path
import sys
import jieba
import re

#################### config ###################
path = "../data/"
# data = "2017-07-27-22-01_QueryQuestions.tsv.toy"
# query_data = "2017-07-27-22-01_Query.tsv.toy"
# question_data = "2017-07-27-22-01_Question.tsv.toy"
data = "2017-07-27-22-01_QueryQuestions.tsv"
query_data = "2017-07-27-22-01_Query.tsv"
question_data = "2017-07-27-22-01_Question.tsv"
############### end of config #################

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

def CreateToyData(count):
    inputfile = path + data
    outputfile = inputfile + str(count)
    
    input = open(inputfile, 'r')
    output = open(outputfile, 'w')

    i = 0
    for line in input.readlines():
        output.write(line)
        i = i + 1
        if (i % 10000 == 0):
            print(i)
        if (i == count):
            break

def SplitQueryQuestion():
    logger.info("running Split Query and Question " + path + data)
    inputfile = path + data
    queryfile = path + query_data
    questionfile = path + question_data
    i = 0
    output_query = open(queryfile, 'w')
    output_question = open(questionfile, 'w')
    input = open(inputfile, 'r')

    for line in input.readlines():
        query, question = line.split('\t')
        output_query.write(query + '\n')
        output_question.write(question)
        i = i + 1
        if (i % 100000 == 0):
            logger.info("Split " + str(i) + " articles")

    output_query.close()
    output_question.close()
    logger.info("Finished Saved " + str(i) + " articles in" + queryfile + " and " + questionfile) 

def Tradition2Simple(data):
    logger.info("running Tradition to Simple in " + path + data)

    inputfile = path + data
    outputfile = path + data + ".zhs"
    cmd = "opencc -i " + inputfile + " -o " + outputfile + " -c zht2zhs.ini"
    os.system(cmd)

def WordBeark(data):
    logger.info("running Word Beark in " + path + data)

    # inputfile = path + data + ".zhs"
    inputfile = path + data
    outputfile = path + data + ".wordbreak"
    i = 0
    output = open(outputfile, 'w')
    input = open(inputfile, 'r')

    for line in input.readlines():
        seg_list = jieba.cut(line)
        output.write(u' '.join(seg_list))

        i = i + 1
        if (i % 10000 == 0):
            logger.info("Cut " + str(i) + " in " + data)

    output.close()
    logger.info("Finished Saved " + str(i) + " articles in " + outputfile)

#CreateToyData(100)

#SplitQueryQuestion()
##########################
# Tradition2Simple will products inconsistent szie between query_data and question_data
# Tradition2Simple(query_data)
# Tradition2Simple(question_data)
##########################
#WordBeark(query_data)
#WordBeark(question_data)

def WordBearkSplit(data, count):
    logger.info("running WordBreak and Split in " + path + data)

    inputfile = path + data
    queryfile = path + query_data + str(count)
    questionfile = path + question_data + str(count)
    i = 0
    output_query = open(queryfile, 'w')
    output_question = open(questionfile, 'w')
    input = open(inputfile, 'r')

    for line in input.readlines():
        query, question = line.split('\t')

        query_list = jieba.cut(query)
        question_list = jieba.cut(question)
        # if (len(query_list) <= 1) or (len(question_list) <= 1):
        #     logger.info("Invalid train data in {}: {}, {}".format(i, query, question))
        #     train_number = train_number - 1
        #     continue

        output_query.write(u' '.join(query_list) + '\n')
        output_question.write(u' '.join(question_list))
        i = i + 1
        if (i % 100000 == 0):
            logger.info("WordBeark and Split " + str(i) + " in " + inputfile)

    output_query.close()
    output_question.close()
    logger.info("Finished Saved " + str(i) + " articles in" + queryfile + " and " + questionfile)

count = 1500000
CreateToyData(count)
WordBearkSplit("2017-07-27-22-01_QueryQuestions.tsv"+str(count), count)