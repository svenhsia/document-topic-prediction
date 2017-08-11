import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os, glob, json
from pymongo import MongoClient

mongoConnection = MongoClient()
trainData = mongoConnection.trainDataNew
documents = trainData.documents
toDel = trainData.delete
documents.drop()
toDel.drop()


if os.path.exists('crawlDocuments/'):
    jsonFiles = glob.glob('crawlDocuments/*.json')
    logging.info('start to read json files')
    for file in jsonFiles: # file is of type str
        logging.info('read %s', file)
        with open(file) as f:
            data = json.load(f)
        for doc in data:
            if len(doc['DETAIL']) > 500:
                documents.insert_one(doc)
            else:
                toDel.insert_one(doc)
        logging.info('finish reading %s', file)
