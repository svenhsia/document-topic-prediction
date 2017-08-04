"""Loads corpus from file and training different models, then store models into files"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import utils
from gensim import corpora, models

CATEGORY = utils.CATEGORY
# CATEGORY_NAME = utils.getCategoryName()
CATEGORY_NAME = 'reference'
# NUMBER_OF_TOPICS = utils.NUMBER_OF_TOPICS
NUMBER_OF_TOPICS = 2
logging.info('start to train models about category ' + CATEGORY_NAME)
logging.info('train models in ' + str(NUMBER_OF_TOPICS) + ' topics')

# TODO: need to be tested
# check md5 of corpus and dictionary file
# if utils.checkMD5('src/{}/corpus/id2word_{}.dict'.format(CATEGORY_NAME, CATEGORY_NAME)) and utils.checkMD5('src/{}/corpus/corpus_{}.mm'.format(CATEGORY_NAME, CATEGORY_NAME)):
#     logging.info('models are latest')
#     exit()

# load corpus and word2id dictionary from disk
if os.path.exists('src/{}/corpus/id2word_{}.dict'.format(CATEGORY_NAME, CATEGORY_NAME)):
    dictionary = corpora.Dictionary.load('src/{}/corpus/id2word_{}.dict'.format(CATEGORY_NAME, CATEGORY_NAME))
    corpus = corpora.MmCorpus('src/{}/corpus/corpus_{}.mm'.format(CATEGORY_NAME, CATEGORY_NAME))
    logging.info("use files generated from getDocuments.py")
else:
    logging.error("Data set was not generated.")
    exit(1)

# Remove existant files
utils.initializeDirectory('src/{}/models'.format(CATEGORY_NAME))

# training Tf-IDF model / initialize a model
logging.info("start to training Tf-idf model")
tfidf = models.TfidfModel(corpus)
tfidf.save('src/{}/models/model_{}.tfidf'.format(CATEGORY_NAME, CATEGORY_NAME))
corpus_tfidf = tfidf[corpus]    # transform corpus into tfidf model
corpora.MmCorpus.serialize('src/{}/corpus/corpus_tfidf_{}.mm'.format(CATEGORY_NAME, CATEGORY_NAME), corpus_tfidf)   # store tfidf model corpus in disk for later use
logging.info("Tf-idf model training finished.\n")

# training LSI model
dimensions = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200]
for i in dimensions:
    NUMBER_OF_TOPICS = i
    logging.info("start to training LSI model with %s topics", str(i))
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUMBER_OF_TOPICS)
    lsi.save('src/{}/models/model_{}_{}.lsi'.format(CATEGORY_NAME, CATEGORY_NAME, NUMBER_OF_TOPICS))    # store model in disk
    logging.info("LSI model training finished.\n")

# training LDA model
# ###########
# logging.info("Start to training LDA model")
# lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=NUMBER_OF_TOPICS, update_every=1, chunksize=100, passes=2)
# lda.save('src/{}/models/model_{}.lda'.format(CATEGORY_NAME, CATEGORY_NAME))    # store model in disk
# logging.info("LDA model training finished.\n")
# ###########

# TODO: need to be tested
# utils.updateMD5('src/{}/corpus/id2word_{}.dict'.format(CATEGORY_NAME, CATEGORY_NAME))
# utils.updateMD5('src/{}/corpus/corpus_{}.mm'.format(CATEGORY_NAME, CATEGORY_NAME))

