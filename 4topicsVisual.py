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
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


db = MongoClient().trainDataNew
col = db.documents

CATEGORY = ['alumina', 'timber', 'cement', 'paper']
NUMBER_SAMPLES = 4000
NUMBER_FEATURES = 10
NUMBER_TOPICS = len(CATEGORY)
COLORPOOL = ['red', 'blue', 'green', 'orange', 'black']
colors = COLORPOOL[:NUMBER_TOPICS]

aluminaList = list(col.find({'TOPIC': {'$in': ['alumina', 'bauxite', 'primary_aluminium']}}))
logging.info("get %s documents about alumina", str(len(aluminaList)))
aluminaSamples = random.sample(aluminaList, NUMBER_SAMPLES)

timberList = list(col.find({'TOPIC': 'timber'}))
logging.info("get %s documents about timber", str(len(timberList)))
timberSamples = random.sample(timberList, NUMBER_SAMPLES)

cementList = list(col.find({'TOPIC': 'cement'}))
logging.info("get %s documents about cement", str(len(cementList)))
cementSamples = random.sample(cementList, NUMBER_SAMPLES)

paperList = list(col.find({'TOPIC': 'paper'}))
logging.info("get %s documents about paper", str(len(paperList)))
paperSamples = random.sample(paperList, NUMBER_SAMPLES)

samples = aluminaSamples + timberSamples + cementSamples + paperSamples
random.shuffle(samples)

category2num = {key: value for value, key in enumerate(CATEGORY)}
for key in ['bauxite', 'primary_aluminium', 'secondary_aluminium']:
    category2num[key] = category2num['alumina']

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

# lda = gensim.models.LdaModel(corpus_tfidf, num_topics=NUMBER_FEATURES, id2word=dictionary, update_every=1, chunksize=100, passes=2)
# vectors = lda[corpus_tfidf]
# y = [category2num[doc['TOPIC']] for doc in samples]
# predict = []
# for vec in vectors:
#     sortedVec = sorted(vec, key=lambda x: -x[1])
#     predict.append(sortedVec[0][0])

# correct = 0
# for r, p in zip(y, predict):
#     if r == p:
#         correct += 1
# lda.print_topics()   
# print('accuracy: ' + str(correct / len(y)))


lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=NUMBER_FEATURES, id2word=dictionary)

vectors = lsi[corpus_tfidf]
vectors = [[ele[1] for ele in doc] for doc in vectors]
vectors_ = normalize(vectors)
X = pd.DataFrame(vectors_)
y = [category2num[doc['TOPIC']] for doc in samples]
y = np.array(y)

# pca = PCA(n_components=2)
# transformed = pd.DataFrame(pca.fit(X).transform(X))
# fig = plt.figure()
# for index in range(NUMBER_TOPICS):
#     plt.scatter(transformed[y == index][0], transformed[y == index][1], label=CATEGORY[index], color=COLORPOOL[index], alpha=0.5)
# plt.title('Clustering of documents by PCA ({} samples, {} features)'.format(NUMBER_SAMPLES, NUMBER_FEATURES))
# plt.legend(loc='best')
# # plt.axis('off')
# plt.show()

pca = PCA(n_components=3)
transformed = pd.DataFrame(pca.fit(X).transform(X))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index in range(NUMBER_TOPICS):
    ax.scatter(transformed[y == index][0], transformed[y == index][1], transformed[y == index][2], label=CATEGORY[index], color=COLORPOOL[index], alpha=0.5)
plt.title('Clustering of documents by PCA ({} samples, {} features)'.format(NUMBER_SAMPLES, NUMBER_FEATURES))
plt.legend(loc='best')
# plt.axis('off')
plt.show()


# pair = [3, 2]
# pca = PCA(n_components=2)
# transformed = pd.DataFrame(pca.fit(X).transform(X))
# fig = plt.figure()
# for index in pair:
#     plt.scatter(transformed[y == index][0], transformed[y == index][1], label=CATEGORY[index], color=COLORPOOL[index], alpha=0.5)
# plt.title('Clustering of documents by PCA ({} samples, {} features)'.format(NUMBER_SAMPLES, NUMBER_FEATURES))
# plt.legend(loc='best')
# # plt.axis('off')
# plt.show()

# pca = PCA(n_components=3)
# transformed = pd.DataFrame(pca.fit(X).transform(X))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for index in pair:
#     ax.scatter(transformed[y == index][0], transformed[y == index][1], transformed[y == index][2], label=CATEGORY[index], color=COLORPOOL[index], alpha=0.5)
# plt.title('Clustering of documents by PCA ({} samples, {} features)'.format(NUMBER_SAMPLES, NUMBER_FEATURES))
# plt.legend(loc='best')
# # plt.axis('off')
# plt.show()



