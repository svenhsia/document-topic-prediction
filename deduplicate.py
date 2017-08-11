import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from pymongo import MongoClient
from bson.objectid import ObjectId

def tokenizer(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def areSameText(text1, text2):
    newline1 = ''.join(text1)
    newline2 = ''.join(text2)
    l1 = len(newline1)
    l2 = len(newline2)
    if l1 > 1.5 * l2 or l2 > 1.5 * l1:
        return False
    pos = 0
    lseg = l1 // 10
    segs = [newline1[s:(s + lseg // 3)] for s in [lseg * i for i in range(10)]]
    for seg in segs:
        if seg in newline2:
            pos += 1
        if pos >= 5:
            return True
    # print(pos, neg)
    return False

connection = MongoClient()
db = connection.trainingDocuments
col = db.documents
dup = db.duplicate

documentList = list(col.find())

pool = []
rest = []

num = 0
for index, doc in enumerate(documentList):
    if index % 500 == 0:
        logging.info("process %s documents", str(index))
    tokens = tokenizer(doc["cleanBody"])
    flag = False
    for i, p in enumerate(pool):
        if areSameText(p, tokens):
            doc1 = {key: val for key, val in doc.items() if key != '_id'}
            doc1['num'] = num
            doc2 = {key: val for key, val in documentList[rest[i]].items() if key != '_id'}
            doc2['num'] = num
            num += 1
            dup.insert_many([doc1, doc2])
            col.delete_one({"_id": ObjectId(doc["_id"])})
            flag = True
            break
    if not flag:
        pool.append(tokens)
        rest.append(index)



