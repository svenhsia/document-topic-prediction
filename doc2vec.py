import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def tokenizer(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

dictionary = gensim.corpora.Dictionary.load('src/reference/corpus/id2word_reference.dict')

# for i in [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200]:

#     lsi = gensim.models.LsiModel.load('src/reference/models/model_reference_{}.lsi'.format(i))

#     # fout = open('src/reference/trainingset_{}.vec'.format(i), 'w')
#     # with open('src/reference/corpus/documents_reference.txt', 'r') as f:
#     #     logging.info('start to get documents vector representation')
#     #     for index, line in enumerate(f):
#     #         if index % 100 == 0:
#     #             logging.info('treat %s documents', str(index))
#     #         doc = tokenizer(line)
#     #         doc_bow = dictionary.doc2bow(doc)
#     #         vec_lsi = lsi[doc_bow]
#     #         vec = [str(ele[1]) for ele in vec_lsi]
#     #         fout.write(' '.join(vec) + '\n')
#     # fout.close()

#     fout = open('src/reference/labelset_{}.vec'.format(i), 'w')
#     with open('src/alumina+timber/corpus/documents_alumina+timber.txt', 'r') as f:
#         logging.info('start to get documents vector representation')
#         for index, line in enumerate(f):
#             if index % 500 == 0:
#                 logging.info('treat %s documents', str(index))
#             doc = tokenizer(line)
#             doc_bow = dictionary.doc2bow(doc)
#             vec_lsi = lsi[doc_bow]
#             vec = [str(ele[1]) for ele in vec_lsi]
#             fout.write(' '.join(vec) + '\n')
#     fout.close()

def doc2vec(num_features, *docs):
    """transform an arbitary number of document texts into vector representation with a specific number of features"""
    lsi = gensim.models.LsiModel.load('src/reference/models/model_reference_{}.lsi'.format(num_features))
    rst = []
    if isinstance(docs[0], list):
        for doc in docs[0]:
            doc_bow = dictionary.doc2bow(tokenizer(doc))
            vec = [ele[1] for ele in lsi[doc_bow]]
            rst.append(vec)
        return rst
    for doc in docs:
        doc_bow = dictionary.doc2bow(tokenizer(doc))
        vec = [ele[1] for ele in lsi[doc_bow]]
        rst.append(vec)
    return rst

