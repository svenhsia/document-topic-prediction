import logging
from gensim.summarization import summarize, keywords
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text = ''

with open('autoSum.txt', 'r') as f:
    for line in f:
        text += line

text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
while '  ' in text:
    text = text.replace('  ', ' ')

print(summarize(text, ratio=0.2))
# print(keywords(text, ratio=0.1))
