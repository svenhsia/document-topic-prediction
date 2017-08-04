"""Extract translated documents and store it into a plain text file, then use gensim to transform it into a corpus file"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import utils

# get CATEGORY and documentList of this category
CATEGORY = utils.CATEGORY
CATEGORY_NAME = utils.getCategoryName()
# CATEGORY_NAME = 'reference'
documentList = utils.getDocumentList(CATEGORY)
logging.info('got %d documents about category %s', len(documentList), CATEGORY_NAME)


# Delete existant files
utils.initializeDirectory('src/{}/corpus/'.format(CATEGORY_NAME))

# write documents into file
corpus_file = open('src/{}/corpus/documents_{}.txt'.format(CATEGORY_NAME, CATEGORY_NAME), 'w')
logging.info("write documents' contents into corpus_file_%s.txt", CATEGORY_NAME)
for doc in documentList:
    corpus_file.write(doc["DETAIL"] + "\n")
corpus_file.close()

# # TODO: need to be tested
# # check md5 of corpus_file.txt to determine whether calculate corpus and dictionary or not
# # if utils.checkMD5('src/{}/corpus/corpus_file_{}.txt'.format(CATEGORY_NAME, CATEGORY_NAME)): # md5 matches, no need to update
# #     logging.info('corpus file and dictionary are latest')
# #     exit()


# # TODO: need to be tested
# # utils.updateMD5('src/{}/corpus/corpus_file_{}.txt'.format(CATEGORY_NAME, CATEGORY_NAME))    # update md5
