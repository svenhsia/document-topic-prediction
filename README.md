# Documents Topic Detection

## INTRODUCTION

The object of this project is to train a model to detect the topic of a given text. In the domain of automatic e-reporting and e-reputation detection, we expect that the documents we grabbed from Internet are topic concentrated. For exemple, we want to grab news about primary material cobalt industry. However, documents grabbed by key word "cobalt" may actually talk about *Cobalt Air*.
By using Topic Modeling method to vectorize a natural language text, we could operate a semantic analysis and quantify text documents similarity. Thus, we could build a classifier to detect a document's topic, so that we could filter documents before using them for following work.


## How to play with

1. Make sure that your machine has python 3.5+ environment. Packages **gensim** and **Scikit-learn** are necessary to run this programme.

2. Copy the document whose topic you want to detect into the file **todetect.txt** under the root directory of project. The text have to be in English, and no preprocessing is required.

3. Run the script **topicDetection.py** under the same directory. The prediction result will be printed in the standard out.

## Issues

1. Due to the lack of labeled documents, the model can only project a document into one of these 4 topics cited above. Even if the document provided has nothing with these topics, it will still be predicted as one of these topics.

## TODO

1. Add more labeled documents about more topics to enlarge the topic diversity of model.

2. Add the fonctionality to predict a document as "UNKNOWN" when the document has a weak similarity to all topics existing in the model.
