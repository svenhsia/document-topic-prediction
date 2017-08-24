import logging
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def tokenizer(doc_line):
    return [token for token in simple_preprocess(doc_line) if token not in STOPWORDS]

text = ''
with open('todetect.txt', 'r') as f:
    for line in f:
        text = text + line
text = text.replace('\n', ' ').replace('\r', ' ')
doc = tokenizer(text)

dictionary = corpora.Dictionary.load('src/word2id.dict')
tfidf = models.TfidfModel.load('src/model.tfidf')
lsi = models.LsiModel.load('src/model.lsi')

corpus_test = dictionary.doc2bow(doc)
corpus_tf = tfidf[corpus_test]
corpus_lsi = lsi[corpus_tf]
vec_lsi = [t[1] for t in corpus_lsi]
vec_norm = normalize([vec_lsi])[0]


# clf = joblib.load('src/svm.pkl')
clf = joblib.load('src/nn.pkl')
# clf = joblib.load('src/knn.pkl')
# clf = joblib.load('src/rf.pkl')
y_predict = clf.predict([vec_norm])[0]
label_dict = {0:'alumina', 1: 'timber', 2: 'cement', 3: 'paper'}
label = label_dict[y_predict]
print(label)
