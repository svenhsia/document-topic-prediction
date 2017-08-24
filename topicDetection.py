import logging
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def tokenizer(doc_line):
    return [token for token in simple_preprocess(doc_line) if token not in STOPWORDS]

print("We provide 4 classifiers to detect the document's topic:\nNeural Network\tSupport Vector Machine\tK Nearest Neighbors\tRandom Forest\n")
clf = input("Choose the classifier you want to use by input one of NN, SVM, KNN and RF:")
clf_name = {"NN": "Neural Network", "SVM": "Support Vector Machine", "KNN": "K Nearest Neighbors", "RF": "Random Forest"}

while clf not in clf_name:
    clf = input("\nPlease choose from NN, SVM, KNN and RF:")

print("\nYou have choosen {}.".format(clf_name[clf]))

text = ''
doc_input = input("\nPlease input your document as a single line string,\nor press ENTER if you have copied it into todetect.txt:")

if doc_input == '':
    with open('todetect.txt', 'r') as f:
        for line in f:
            text = text + line
    text = text.replace('\n', ' ').replace('\r', ' ')
else:
    text = doc_input

doc = tokenizer(text)

dictionary = corpora.Dictionary.load('src/word2id.dict')
tfidf = models.TfidfModel.load('src/model.tfidf')
lsi = models.LsiModel.load('src/model.lsi')
clf = joblib.load('src/{}.pkl'.format(clf))

corpus_test = dictionary.doc2bow(doc)
corpus_tf = tfidf[corpus_test]
corpus_lsi = lsi[corpus_tf]
vec_lsi = [t[1] for t in corpus_lsi]
vec_norm = normalize([vec_lsi])[0]

y_predict = clf.predict([vec_norm])[0]
label_dict = {0:'alumina', 1: 'timber', 2: 'cement', 3: 'paper'}
label = label_dict[y_predict]
print('\nYour document talks about "{}".\n'.format(label))
