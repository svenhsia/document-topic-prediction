import logging
from collections import defaultdict
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim
from utils import initializeDirectory
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=E0401, W0611
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from utils import computeAccuracy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tokenizer(doc_line):
    return [token for token in simple_preprocess(doc_line) if token not in STOPWORDS]


class TopicDetector(object):
    name = "topic_detector"
    num_topics = 0
    labels_literal = ()
    labels2num = {}
    labels = []
    num_features = 0
    models_features = []
    nn_layout = ()
    nn_alpha = 0

    def __init__(self, name, labels_literal):
        self.name = name
        self.labels_literal = tuple(labels_literal)
        self.num_topics = len(labels_literal)
        self.labels2num = {l:i for i, l in enumerate(labels_literal)}
        initializeDirectory(name + '/corpus') # TODO
        initializeDirectory(name + '/model')
        initializeDirectory(name + '/matrix')
        initializeDirectory(name + '/clustering')

    def feedDocuments(self, labels, documents, label_in_number=False):
        logging.info("start to feed documents")
        if len(labels) != len(documents):
            raise ValueError("The length of labels and documents do not match.")
        self.labels = labels if label_in_number else [self.labels2num[l] for l in labels]
        texts = [tokenizer(d) for d in documents]
        freq = defaultdict(int)
        for text in texts:
            for token in text:
                freq[token] += 1
        texts = [[token for token in text if freq[token] > 10] for text in texts]

        logging.info("start to train tfidf")
        dictionary = gensim.corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=20, no_above=0.4, keep_n=None)
        dictionary.save(self.name + '/corpus/word2id.dict')
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = gensim.models.TfidfModel(dictionary=dictionary)
        tfidf.save(self.name + '/corpus/model.tfidf')
        corpus_tfidf = tfidf[corpus]
        gensim.corpora.MmCorpus.serialize(self.name + '/corpus/corpus_tfidf.mm', corpus_tfidf)

    def trainLsiModel(self, models_features=None):
        logging.info("start to train lsi model")
        self.models_features = [self.num_topics * i for i in [1, 2, 3, 5, 8]] if models_features is None else models_features
        dictionary = gensim.corpora.Dictionary.load(self.name + '/corpus/word2id.dict')
        corpus = gensim.corpora.MmCorpus(self.name + '/corpus/corpus_tfidf.mm')
        for num_features in self.models_features:
            lsi = gensim.models.LsiModel(corpus, num_topics=num_features, id2word=dictionary, distributed=False)
            lsi.save(self.name + '/model/model_{}.lsi'.format(num_features))
            vectors_tuple = lsi[corpus]
            doc_vec = [[tup[1] for tup in vector_tuple] for vector_tuple in vectors_tuple]
            doc_vec = pd.DataFrame(doc_vec)
            doc_vec.insert(0, 'label', self.labels)
            doc_vec.to_csv(self.name + '/matrix/docvec_{}.csv'.format(num_features), index=False)

    def clusterEval(self, show_scatter_graph=False, show_accuracy_curve=False):
        logging.info("start to clustering")
        methods = ['KMeans', 'Hierarchical']
        bm = pd.DataFrame([[0 for i in range(len(self.models_features))] for j in range(len(methods))], columns=self.models_features, index=methods)
        for method in methods:
            for num_features in self.models_features:
                logging.info("clustering %s %s", method, str(num_features))
                data = pd.read_csv(self.name + '/matrix/docvec_{}.csv'.format(num_features))
                y = data.loc[:, 'label']
                X = data.iloc[:, 1:]
                if method == 'KMeans':
                    X_norm = pd.DataFrame(normalize(X))
                    cluster = KMeans(n_clusters=self.num_topics, n_jobs=-1, n_init=20, max_iter=500, tol=1e-5, precompute_distances=True)
                    cluster.fit(X_norm)
                else:
                    cluster = AgglomerativeClustering(n_clusters=self.num_topics, affinity="cosine", linkage='average')
                    cluster.fit(X)
                y_pred = np.array(cluster.labels_)
                y_array = np.array(y)
                contingency_table = pd.DataFrame([[0 for i in range(self.num_topics)] for j in range(self.num_topics)], columns=self.labels_literal, index=self.labels_literal)
                for pred_l in range(self.num_topics):
                    for true_l in range(self.num_topics):
                        contingency_table.iloc[pred_l, true_l] = sum((y_pred == pred_l) * (y_array == true_l))
                contingency_table.to_csv(self.name + '/clustering/ct_{}_{}.csv'.format(method, num_features))
                accuracy = computeAccuracy(contingency_table)
                bm.loc[method, num_features] = accuracy
        bm.to_csv(self.name + '/clustering/comparaison.csv')
        find_flag = False
        clustering_threshold = 0.90
        while not find_flag:
            clustering_threshold -= 0.02
            for num_features, all_methods in bm.iteritems():
                if all_methods.max() > clustering_threshold:
                    find_flag = True
                    self.num_features = num_features
                    break

        if show_accuracy_curve:
            plt.figure()
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            lineStyles = ['o-', '^--', 'd-.', '*:', '+-', 'x--']
            counter = 0
            for method, accuracies in bm.iterrows():
                plt.plot(accuracies.index, accuracies, lineStyles[counter], color=colors[counter], label=method)
                counter += 1
            plt.legend(loc='best')
            plt.xlabel('dimension of topic models')
            plt.ylabel('accuracy')
            plt.show()

        if show_scatter_graph:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            num_samples = len(self.labels) // 4
            samples = random.sample(range(len(self.labels)), num_samples)
            data = pd.read_csv(self.name + '/matrix/docvec_{}.csv'.format(self.num_features))
            y = np.array(data.loc[samples, 'label'])
            X = normalize(data.iloc[samples, 1:])
            pca = PCA(n_components=3)
            transformed = pca.fit(X).transform(X)
            transformed = pd.DataFrame(transformed)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for index in range(self.num_topics):
                dots_to_plot = transformed[y == index]
                ax.scatter(dots_to_plot[0], dots_to_plot[1], dots_to_plot[2], label=self.labels_literal[index], color=colors[index], alpha=0.5)
            plt.legend(loc='best')
            plt.show()


    def trainNN(self, show_graph=False, eval_other_models=False):
        # alphas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        alphas = [1e-5, 1e-3, 0.1, 10]
        models = self.models_features if eval_other_models else [self.num_features]
        bm = pd.DataFrame([[0 for a in alphas] for m in models], columns=alphas, index=models)
        for num_features in models:
            data = pd.read_csv(self.name + '/matrix/docvec_{}.csv'.format(num_features))
            y = data.loc[:, 'label']
            X = data.iloc[:, 1:]
            X = normalize(X)
            X, y = shuffle(X, y)

            for alpha in alphas:
                clf = MLPClassifier(hidden_layer_sizes=(num_features,), activation="logistic", solver='lbfgs', alpha=alpha, max_iter=500)
                accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
                bm.loc[num_features, alpha] = accuracy
        bm.to_csv(self.name + '/nn_accuracy_alpha.csv')
        find_flag = False
        accuracy_threshold = 0.98
        while not find_flag:
            accuracy_threshold -= 0.01
            for best_alpha, all_models in bm.iteritems():
                if all_models.max() > accuracy_threshold:
                    find_flag = True
                    self.num_features = all_models.idxmax()
                    self.nn_alpha = best_alpha
                    self.nn_layout = (self.num_features,)
                    break
        
        if show_graph:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            lineStyles = ['o-', '^--', 'd-.', '*:', '+-', 'x--']
            counter = 0
            for model, accuracies in bm.iterrows():
                plt.plot(accuracies.index, accuracies, lineStyles[counter], color=colors[counter], label=model)
                counter += 1
            plt.legend(loc='best')
            plt.xlabel('alpha')
            plt.ylabel('accuracy')
            plt.show()


    def plotLearningCurve(self):
        pass

    def saveClassifier(self):
        clf = MLPClassifier(hidden_layer_sizes=self.nn_layout, activation="logistic", solver='lbfgs', alpha=self.nn_alpha, max_iter=500)
        data = pd.read_csv(self.name + '/matrix/docvec_{}.csv'.format(self.num_features))
        y = data.loc[:, 'label']
        X = data.iloc[:, 1:]
        X = normalize(X)
        clf.fit(X, y)
        joblib.dump(clf, self.name + '/nn.pkl')

    def predictLabel(self, to_detect=None, add_to_model=0):
        dictionary = gensim.corpora.Dictionary.load(self.name + '/corpus/word2id.dict')
        tfidf = gensim.models.TfidfModel.load(self.name + '/corpus/model.tfidf')
        lsi = gensim.models.LsiModel.load(self.name + '/model/model_{}.lsi'.format(self.num_features))
        clf = joblib.load(self.name + '/nn.pkl')
        doc = tokenizer(to_detect)
        corpus_test = dictionary.doc2bow(doc)
        corpus_tfidf = tfidf[corpus_test]
        corpus_lsi = lsi[corpus_tfidf]
        vec_lsi = [t[1] for t in corpus_lsi]
        vec_norm = normalize([vec_lsi])[0]

        y_predict = clf.predict([vec_norm])[0]
        return self.labels_literal[y_predict]
