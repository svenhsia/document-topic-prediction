import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import random
from functools import reduce
from sklearn.neural_network import MLPClassifier
import utils

CATEGORY = utils.CATEGORY
# CATEGORY_NAME = utils.getCategoryName()
CATEGORY_NAME = 'reference'
NUMBER_OF_TOPICS = len(CATEGORY)
documentLists = [utils.getDocumentList([categ]) for categ in CATEGORY]  # lists of documents of different category
for docl in documentLists:
    random.shuffle(docl)
SIZE_TRAIN = 1000
trainDoc = reduce((lambda x, y: x + y), [docl[:SIZE_TRAIN] for docl in documentLists])
validDoc = reduce((lambda x, y: x + y), [docl[SIZE_TRAIN:] for docl in documentLists])
logging.info("randomly composse training dataset if size %s and cross-validation dataset of size %s", str(len(trainDoc)), str(len(validDoc)))
trainTopics = [doc['TOPIC'] for doc in trainDoc]
validTopics = [doc['TOPIC'] for doc in validDoc]
NUMBER_OF_FEATURES = 20
# COLORPOOL = ['red', 'blue', 'green', 'yello', 'orange', 'purple', 'black']

category2num = {key: value for value, key in enumerate(CATEGORY)}

trainY = [category2num[doc['TOPIC']] for doc in trainDoc]
validY = [category2num[doc['TOPIC']] for doc in validDoc]

docVectors = []
with open('src/{}/labelset_{}.vec'.format(CATEGORY_NAME, NUMBER_OF_FEATURES)) as f:
    for line in f:
        vector = [float(i) for i in line.split()]
        docVectors.append(vector)

trainX = [docVectors[doc['num']] for doc in trainDoc]
validX = [docVectors[doc['num']] for doc in validDoc]

clf = MLPClassifier(hidden_layer_sizes=(NUMBER_OF_FEATURES,))
clf.fit(trainX, trainY)

count_rst = [[0 for i in range(NUMBER_OF_TOPICS)] for j in range(NUMBER_OF_TOPICS)]
predictY = clf.predict(validX)

for r, p in zip(validY, predictY):
    count_rst[p][r] += 1

total = len(validY)
correct = sum([count_rst[i][i] for i in range(NUMBER_OF_TOPICS)])
accuracy = correct / total
print(count_rst)
print("accuracy = " + str(accuracy))

