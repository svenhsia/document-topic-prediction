import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from pymongo import MongoClient
import random
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=E0401, W0611
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


db = MongoClient().trainDataNew
col = db.documents

CATEGORY = ['alumina', 'timber', 'cement', 'paper']
NUMBER_SAMPLES = 4000
NUMBER_FEATURES = 10
NUMBER_TESTS = 600
NUMBER_TOPICS = len(CATEGORY)
COLORPOOL = ['red', 'blue', 'green', 'orange', 'black']
colors = COLORPOOL[:NUMBER_TOPICS]

aluminaList = list(col.find({'TOPIC': {'$in': ['alumina', 'bauxite', 'primary_aluminium']}}))
logging.info("get %s documents about alumina", str(len(aluminaList)))
random.shuffle(aluminaList)
aluminaSamples = aluminaList[:NUMBER_SAMPLES]
aluminaTest = aluminaList[NUMBER_SAMPLES:NUMBER_SAMPLES + NUMBER_TESTS] if len(aluminaList) >= NUMBER_SAMPLES + NUMBER_TESTS else aluminaList[NUMBER_SAMPLES:]

timberList = list(col.find({'TOPIC': 'timber'}))
logging.info("get %s documents about timber", str(len(timberList)))
random.shuffle(timberList)
timberSamples = timberList[:NUMBER_SAMPLES]
timberTest = timberList[NUMBER_SAMPLES:NUMBER_SAMPLES + NUMBER_TESTS] if len(timberList) >= NUMBER_SAMPLES + NUMBER_TESTS else timberList[NUMBER_SAMPLES:]

cementList = list(col.find({'TOPIC': 'cement'}))
logging.info("get %s documents about cement", str(len(cementList)))
random.shuffle(cementList)
cementSamples = cementList[:NUMBER_SAMPLES]
cementTest = cementList[NUMBER_SAMPLES:NUMBER_SAMPLES + NUMBER_TESTS] if len(cementList) >= NUMBER_SAMPLES + NUMBER_TESTS else cementList[NUMBER_SAMPLES:]

paperList = list(col.find({'TOPIC': 'paper'}))
logging.info("get %s documents about paper", str(len(paperList)))
random.shuffle(paperList)
paperSamples = paperList[:NUMBER_SAMPLES]
paperTest = paperList[NUMBER_SAMPLES:NUMBER_SAMPLES + NUMBER_TESTS] if len(paperList) >= NUMBER_SAMPLES + NUMBER_TESTS else paperList[NUMBER_SAMPLES:]

samples = aluminaSamples + timberSamples + cementSamples + paperSamples
test = aluminaTest + timberTest + cementTest + paperTest


category2num = {key: value for value, key in enumerate(CATEGORY)}
for key in ['bauxite', 'primary_aluminium', 'secondary_aluminium']:
    category2num[key] = category2num['alumina']

random.shuffle(samples)



def tokenizer(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

texts = [tokenizer(doc['DETAIL']) for doc in samples]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=NUMBER_FEATURES, id2word=dictionary)

vectors = lsi[corpus_tfidf]
vectors = [[ele[1] for ele in doc] for doc in vectors]
X = normalize(vectors)
y = [category2num[doc['TOPIC']] for doc in samples]
y = np.array(y)


# test samples
logging.info("start to get test samples")
y_test = [category2num[doc['TOPIC']] for doc in test]
X_test = []
for testdoc in test:
    doctext = testdoc['DETAIL']
    doc_bow = dictionary.doc2bow(tokenizer(doctext))
    doc_vec = [ele[1] for ele in lsi[doc_bow]]
    X_test.append(doc_vec)
X_test = normalize(X_test)

kpool = [1, 2, 5, 10, 20, 30, 50, 80, 100, 120, 150, 200, 250, 300, 350, 400]
train_scores = []
test_scores = []
for k in kpool:
    logging.info("train KNN with k=%s", str(k))
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X, y)

    train_score = knn.score(X, y)
    test_score = knn.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(kpool, train_scores, 'o-', label='train_score')
plt.plot(kpool, test_scores, 'g^-', label='test_score')
plt.title('KNN classification k-curve')
plt.legend(loc='best')
plt.show()



