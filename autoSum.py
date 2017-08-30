import logging
import re
import networkx as nx
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import matplotlib.pyplot as plt
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

fraction = 0.2


def tokenizer(doc_line):
    return [token for token in simple_preprocess(doc_line) if token not in STOPWORDS]

filename = 'autoSum.txt'
f = open(filename)
text = ''
with open(filename, 'r') as f:
    for line in f:
        text = text + line
text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
while '  ' in text:
    text = text.replace('  ', ' ')
sent_str = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)


dictionary = corpora.Dictionary.load('src/word2id.dict')
tfidf = models.TfidfModel.load('src/model.tfidf')
lsi = models.LsiModel.load('src/model.lsi')

sent_corpus = [tokenizer(s) for s in sent_str]
sent_corpus = [dictionary.doc2bow(line) for line in sent_corpus]
sent_tfidf = tfidf[sent_corpus]
sent_lsi = lsi[sent_tfidf]
sent_vec = [[t[1] for t in sent] for sent in sent_lsi]

sim_matrix = cosine_similarity(sent_vec, sent_vec)

n_sentences = len(sent_vec)

G = nx.Graph()
for i in range(n_sentences):
    for j in range(i + 1, n_sentences):
        G.add_edge(i, j, weight=sim_matrix[i][j])

pr = nx.pagerank(G, alpha=0.85, max_iter=500)

sentences_sorted = sorted(pr.items(), key=lambda x: -x[1])

head_length = round(len(sentences_sorted) * fraction) if fraction < 1 else int(fraction)

head_sentences = []
for i in range(head_length):
    sent_index = sentences_sorted[i][0]
    head_sentences.append(sent_index)
head_sentences.sort()

for i in head_sentences:
    print(sent_str[i])
