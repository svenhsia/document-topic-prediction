import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import random
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=E0401, W0611
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


CATEGORY = utils.CATEGORY
# CATEGORY_NAME = utils.getCategoryName()
CATEGORY_NAME = 'reference'
NUMBER_OF_TOPICS = utils.NUMBER_OF_TOPICS
documentList = utils.getDocumentList(CATEGORY)
logging.info('got %s documents', str(len(documentList)))
TOPICS = [doc['TOPIC'] for doc in documentList]
SITES = [doc['SITE'] for doc in documentList]
WEBSITES = ['AlCircle', 'Timberbiz', 'TimberProcessing']
NUMBER_OF_FEATURES = 20
NUMBER_OF_SAMPLES = 1000
COLORPOOL = ['red', 'blue', 'green', 'yello', 'orange', 'purple', 'black']

docVectors = []
with open('src/{}/labelset_{}.vec'.format(CATEGORY_NAME, NUMBER_OF_FEATURES)) as f:
    for line in f:
        vector = [float(i) for i in line.split()]
        docVectors.append(vector)

samples = random.sample(range(len(docVectors)), NUMBER_OF_SAMPLES)


TOPICS_ = [TOPICS[i] for i in samples]
docVectors_ = [docVectors[i] for i in samples]  # pylint: disable=E1126
docVectors_ = normalize(docVectors_)
y = []
for t in TOPICS_:
    for i, c in enumerate(CATEGORY):
        if t == c:
            y.append(i)
            break
y = np.array(y)
colors = COLORPOOL[:NUMBER_OF_TOPICS]


###### visualize with 2 first components ####
plt.figure()
reduced = pd.DataFrame(docVectors_)
for index in range(NUMBER_OF_TOPICS):
    plt.scatter(reduced[y == index][0], reduced[y == index][1], label=CATEGORY[index], color=colors[index], alpha=0.5)
plt.title('Clustering of documents with 2 first components ({} samples, {} features)'.format(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES))
plt.legend(loc='best')
plt.show()


###### visualize in 2D plot #####
pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit(docVectors_).transform(docVectors_))
plt.figure()
for index in range(NUMBER_OF_TOPICS):
    plt.scatter(transformed[y == index][0], transformed[y == index][1], label=CATEGORY[index], color=colors[index], alpha=0.5)
plt.title('Clustering of documents by PCA ({} samples, {} features)'.format(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES))
plt.legend(loc='best')
plt.show()


##### visualize PCA in 3D plot #####
pca = PCA(n_components=3)
transformed = pd.DataFrame(pca.fit(docVectors_).transform(docVectors_))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index in range(NUMBER_OF_TOPICS):
    ax.scatter(transformed[y == index][0], transformed[y == index][1], transformed[y == index][2], label=CATEGORY[index], color=colors[index], alpha=0.5)
plt.title('Clustering of documents by PCA ({} samples, {} features)'.format(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES))
plt.legend(loc='best')
# plt.axis('off')
plt.show()


##### visualize with LDA dimentionality reduction #####
lda = LinearDiscriminantAnalysis(n_components=2)
transformed2 = lda.fit(docVectors_, y).transform(docVectors_)
plt.figure()
for index in range(NUMBER_OF_TOPICS):
    plt.scatter(transformed[y == index][0], transformed[y == index][1], label=CATEGORY[index], color=colors[index], alpha=0.5)
plt.title('Clustering of documents by LDA ({} samples, {} features)'.format(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES))
plt.legend(loc='best')
plt.show()

##### visualize LDA in 3D plot #####
lda = LinearDiscriminantAnalysis(n_components=3)
transformed2 = lda.fit(docVectors_, y).transform(docVectors_)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index in range(NUMBER_OF_TOPICS):
    ax.scatter(transformed[y == index][0], transformed[y == index][1], transformed[y == index][2], label=CATEGORY[index], color=colors[index], alpha=0.5)
plt.title('Clustering of documents by LDA ({} samples, {} features)'.format(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES))
plt.legend(loc='best')
plt.show()

