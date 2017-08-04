"""This module provides some useful functions"""
import logging
import os, glob
import random
import json
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient

# CATEGORY = []
# CATEGORY = ['timber', 'cobalt', 'virtualcontainertech']
# CATEGORY = ['alumina']
CATEGORY = ['alumina', 'timber']
# CATEGORY = ['timber']
# CATEGORY = ['cobalt']
# CATEGORY = ['downstream_products']
# CATEGORY = ['virtualcontainertech']

NUMBER_OF_TOPICS = 2

def getCategoryName():
    """This function transform the category list into a string for use of folder name"""
    logging.info('get category name')
    if len(CATEGORY) == 0:
        return 'all'
    else:
        CATEGORY_copy = CATEGORY
        CATEGORY_copy.sort()
        return "+".join(CATEGORY_copy)

def initializeDirectory(file_path):
    """This function creates a directory according to the file path passed. If the directory already exists, this function deletes all contents inside."""
    logging.info('initialize ' + file_path)
    if os.path.exists(file_path):
        old_files = glob.glob(file_path + '/*')
        for f in old_files:
            os.remove(f)
    else:
        try:
            os.makedirs(file_path)
        except OSError:
            if not os.path.isdir(file_path):
                raise

def getDocumentList(category=None):
    """This function establishes a connecxion to mongodb server and get documents from preselected database"""
    mongoConnection = MongoClient()
    traindb = mongoConnection.trainData
    tigerDocuments = traindb.documents
    logging.info('get access to Mongodb, start to extract documents')
    if category is None or len(category) == 0:
        documentList = list(tigerDocuments.find().sort('num', pymongo.ASCENDING))
    else:
        documentList = list(tigerDocuments.find({'TOPIC': {"$in": category}}).sort('num', pymongo.ASCENDING))
    return documentList

def getDocumentRef():
    """This function establishes a connecxion to mongodb server and get documents of the reference collection"""
    mongoConnection = MongoClient()
    traindb = mongoConnection.trainData
    tigerDocuments = traindb.reference
    logging.info('get access to Mongodb, start to extract documents')
    return list(tigerDocuments.find())


def getDocumentListFromOrigin(category=None):
    """This function establishes a connecxion to mongodb server and extract traslated documents from the original database"""
    mongoConnection = MongoClient()
    tigerdb = mongoConnection.tiger
    tigerDocuments = tigerdb.documents
    logging.info('get access to Mongodb, start to extract documents')
    if category is None or len(category) == 0:
        documentList = list(
            tigerDocuments.aggregate([
                {"$match": {"yandexTranslationBody": {"$exists": True}}},
                {"$match": {"yandexTranslationBody": {"$ne": ""}}},
                {"$match": {"yandexTranslationBody": {"$ne": "Cannot fetch."}}}
            ], allowDiskUse=True))
    else:
        documentList = list(
            tigerDocuments.aggregate([
                {"$match": {"yandexTranslationBody": {"$exists": True}}},
                {"$match": {"yandexTranslationBody": {"$ne": ""}}},
                {"$match": {"yandexTranslationBody": {"$ne": "Cannot fetch."}}},
                {"$match": {"feedCategory": {"$in": category}}}
            ], allowDiskUse=True))
    wantedKeys = ['_id', 'feedCategory', 'yandexTranslationBody']
    return  [{key: doc[key] for key in wantedKeys} for doc in documentList]

def plotHist(inlist):
    """This function plots a DataFrame like dataset as a histogram."""
    df = pd.DataFrame(inlist)
    df.dropna(axis=0, how='all', inplace=True)
    df.plot(kind='bar')
    plt.show()

def plotPie(inlist):
    """This function plots a Serial like dataset as a pie-gram."""
    series = pd.Series(inlist)
    series.dropna(inplace=True)
    series = series[[i for i in series.index if series[i] > 0]]
    series.plot(kind='pie', autopct='%.2f')
    plt.show()

def checkMD5(filename):
    if not os.path.exists(filename):
        logging.error("needed file doesn't exists")
        exit(1)
    newMD5 = hashlib.md5(open(filename, 'rb').read()).hexdigest()
    name = os.path.split(filename)[1]
    with open('checksum.json', 'r+') as f:
        jsonlist = json.load(f)
        checkSum = jsonlist if jsonlist else {}
    if name not in checkSum:    # first time, return false to allow to continue
        return False
    oldMD5 = checkSum[name]
    if oldMD5 != newMD5:
        f.close()
        return False
    else:
        f.close()
        return True

def updateMD5(filename):
    if not os.path.exists(filename):
        logging.error("needed file doesn't exists")
        exit(1)
    newMD5 = hashlib.md5(open(filename, 'rb').read()).hexdigest()
    name = os.path.split(filename)[1]
    f = open('checksum.json', 'r+')
    jsonlist = json.load(f)
    checkSum = jsonlist if jsonlist else {}
    checkSum[name] = newMD5
    f.seek(0)
    json.dump(checkSum, f, ensure_ascii=False)
    f.close()


def cosSim(list1, list2, normalized=True):
    """calculate cosine similarity between two vectors"""
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    len1 = np.linalg.norm(vec1) if not normalized else 1.0
    len2 = np.linalg.norm(vec2) if not normalized else 1.0
    return np.inner(vec1, vec2) / (len1 * len2)

def cosSimV(l, lists, normalized=True):
    """calculate cosine similarities between a vector and vectors in a list"""
    m = np.array(lists)
    v = np.array(l)
    # print(m)
    # print(m.shape)
    mult = np.dot(m, v)
    vlen = np.linalg.norm(v) if not normalized else 1.0
    # print(mult)
    ls = np.linalg.norm(m, axis=1) if not normalized else 1.0
    # print(ls)
    return (mult / (ls * vlen)).tolist()


def randomInitial(end, num, start=0):
    """randomly choose a number of points as an initialisation"""
    pool = list(range(start, end))
    return random.sample(pool, num)


def normalizeVector(vector):
    v = np.array(vector)
    lv = np.linalg.norm(v)
    return (v / lv).tolist()

def normalizeVectors(vectors):
    m = np.array(vectors)
    lens = np.linalg.norm(m, axis=1).reshape(len(vectors), 1)
    return (m / lens).tolist()

def findCentroid(inlists):
    if len(inlists) == 1:
        rst = inlists[0]
    elif not inlists:
        raise Exception("inlists is null")
    rst = np.mean(inlists, axis=0).tolist()
    if isinstance(rst, list):
        print(len(inlists))
        # print(len(inlists[0]))
        print(rst)
        exit(6)
    return rst

