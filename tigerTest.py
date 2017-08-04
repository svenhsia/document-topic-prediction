import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import random
from functools import reduce
from sklearn.neural_network import MLPClassifier
import utils
from doc2vec import doc2vec

CATEGORY = utils.CATEGORY
# CATEGORY_NAME = utils.getCategoryName()
CATEGORY_NAME = 'reference'
NUMBER_OF_TOPICS = len(CATEGORY)

documentLists = [utils.getDocumentList([categ]) for categ in CATEGORY]  # lists of documents of different category
for docl in documentLists:
    random.shuffle(docl)
SIZE_TRAIN = 1200
trainDoc = reduce((lambda x, y: x + y), [docl[:SIZE_TRAIN] for docl in documentLists])
trainTopics = [doc['TOPIC'] for doc in trainDoc]
NUMBER_OF_FEATURES = 20

category2num = {key: value for value, key in enumerate(CATEGORY)}

trainY = [category2num[doc['TOPIC']] for doc in trainDoc]

docVectors = []
with open('src/{}/labelset_{}.vec'.format(CATEGORY_NAME, NUMBER_OF_FEATURES)) as f:
    for line in f:
        vector = [float(i) for i in line.split()]
        docVectors.append(vector)

trainX = [docVectors[doc['num']] for doc in trainDoc]

clf = MLPClassifier(hidden_layer_sizes=(NUMBER_OF_FEATURES,))
clf.fit(trainX, trainY)
logging.info('trained a neural network with %s training data', str(SIZE_TRAIN))

documentList = utils.getDocumentRef()
# COLORPOOL = ['red', 'blue', 'green', 'yello', 'orange', 'purple', 'black']
logging.info('get %s test docs', str(len(documentList)))
Y = [category2num[doc['TOPIC']] for doc in documentList]

docs = [doc['DETAIL'] for doc in documentList]
X = doc2vec(NUMBER_OF_FEATURES, docs)
logging.info('get %s test docs vectors', str(len(X)))

count_rst = [[0 for i in range(NUMBER_OF_TOPICS)] for j in range(NUMBER_OF_TOPICS)]
predictY = clf.predict(X)

logging.info('count accuracy')
for r, p in zip(Y, predictY):
    count_rst[p][r] += 1

total = len(Y)
correct = sum([count_rst[i][i] for i in range(NUMBER_OF_TOPICS)])
accuracy = correct / total
print(count_rst)
print("accuracy = " + str(accuracy))

