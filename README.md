# TIGER Semantic Analysis

## INTRODUCTION
The object of this project is to train a model to predict the topic of a given text. In the domain of automatic e-reporting and e-reputation detection, we expect that the documents we grabbed from Internet are topic concentrated. For exemple, we want to grab news articles about primary material cobalt industry. However, documents grabbed by key word "cobalt" may actually talk about *Cobalt Air*.

By using Topic Modeling method to vectorize a natural language text, we could operate a semantic analysis and quantify text documents similarity. Thus, we could build a neural network to classify a document's topic, so that we could filter documents before using them for following work.


## ARCHITECTURE
This project has 5 parts:
1. Training reference: We use an existant database of more than 20,000 documents grabbed from Internet as a **reference dataset**. These documents are grabbed with key words "alumina" and "timber". We assume that these documents contain enough "knowledges" about these two industry domains. Then we use **gensim** package to train a topic model with a specific number of topics. The number of topics depends on the quality of documents. This topic model provides a corrdinate system with the same number of axis as the number of topics we choose.
2. Get labeled documents: We grab about 7 thousand of documents from somme professionnal websites [AlCircle](http://www.alcircle.com/), [timberbiz](https://www.timberbiz.com.au/) and [timberprocessing](http://www.timberprocessing.com). As these 2 sites focus respectively on the domains of alumina and timber, we could assume that documents from these sites are well labeled.
3. Unsupervised clustering: In order to check accuracy of our model, we vectorize all *labeled* documents with the topic model and plot them out. From the graph we can see clearly the boundary between these 2 topics documents.
![Alt](/src/graph/20features_labelset_norm_3d.png)
The above graph is plotted with normalized vectors. If we plot with original ones, we can even see the orthogonalite of the surfaces composed by documents
![Alt](/src/graph/20features_labelset_3d.png)
4. Train prediction neural network: We train a single-layer neural network with about 2,000 *labeled* documents (1,000 for each topic). In order to evaluate the prediction model, we operate test with other labeled documents. It seems that the accuracy of prediction is better than 99%. We also test with reference dataset, which are considered as less topic-concentrated. The accuracy of prediction achieves 80%.
5. (**TODO**)Predict unknown document: With the prediction neural network, we classify the unknow document to one of these 2 topics. Otherwise, this cannot guarantee more than that this document is more likely to be in the predicted class compared with the other candidate classes. In order to determine whether it belongs to this topic, we calculate the average similarity of this document to all other documents of this topic in the training dataset, and compare this average similarity with a predefined threshold (could be average similarity of all training documents one-to-another) and give a Yes/No decision.

PS: The algorithm is trival. In fact, the main effort are devoted to data collection and data cleaning, because it is the quality of reference dataset and training dataset that determines the quality of model.
