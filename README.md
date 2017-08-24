# Documents Topic Detection

## INTRODUCTION
The object of this project is to train a model to detect the topic of a given text. In the domain of automatic e-reporting and e-reputation detection, we expect that the documents we grabbed from Internet are topic concentrated. For exemple, we want to grab news about primary material cobalt industry. However, documents grabbed by key word "cobalt" may actually talk about *Cobalt Air*.
By using Topic Modeling method to vectorize a natural language text, we could operate a semantic analysis and quantify text documents similarity. Thus, we could build a classifier to detect a document's topic, so that we could filter documents before using them for following work.


## ARCHITECTURE
This project has 5 parts:

1. Get labeled documents: 

    We grab about 7 thousand of documents from somme professionnal websites [AlCircle](http://www.alcircle.com/), [timberbiz](https://www.timberbiz.com.au/), [timberprocessing](http://www.timberprocessing.com), [worldcement.com](https://www.worldcement.com/news/), [globalcement](http://www.globalcement.com/news/) [cemnet](https://www.cemnet.com/News/) and [PaperAge](http://www.paperage.com/). As these sites focus respectively on the domains of *alumina*, *timber*, *cement* and *paper*, we could assume that documents from these sites are well labeled.

2. Training reference: 

    We use an existant database (16,000 documents for now, 4,000 documents for each topic) grabbed from Internet as a **reference dataset**. These documents are grabbed with key words "alumina news", "timber news", "cement news" and "paper industry news". We assume that these documents contain enough "knowledge" about these four industry domains. Then we use **gensim** package to train a topic model with a specific number of features. The number of features is a parameter which we will try to optimize. This topic model provides a corrdinate system with the same number of axis as the number of features we choose, i.e. if the model is trained with 20 features, them each document is vectorized as a vector of length 20.

3. Unsupervised clustering: 

    In order to check the rationality of our model, we vectorize some randomly choosen labeled documents with the topic model and plot them out. From the graph we can see clearly the clusters of documents of different topics.
    ![Alt](/src/4topics/graph/clustering_scatter.png)
    The "distance" between documents are defined by "cosine similarity", which is the inner product of L2 normalized vectors of the documents. This similarity could be considered approximately equalvalent to the Euclidean distance of normalized vectors.
    ![Alt](/src/4topics/graph/clustering_scatter_sphere.png)

4. Train classifiers: 

    We apply 4 commonly used classification algorithms: **K Nearest Neighbour(KNN)**, **Support Vector Machine(SVM)**, **Neural Network(NN)** and **Random Forest(RF)**. For each classifier, we optimize its parameters by cross-validation and gridsearch, and take the optimized model.
    Then we compare these 4 models and choose the most convenient one. Model complexity, computational cost and model accuracy should be considered during the evaluation.

5. Predict unknown document: 

    Use the classifier with best performance to predict the label of an unknown document,  and project the label to topic name among *alumina*, *timber*, *cement* and *paper*.


## Evaluation

1. Clustering accuracy

    One measure to evaluate the accuracy of clustering is to observe the distribution of predict clusters in label classes and the distribution of label classes in predict clusters, as follows:
    ![Alt](/src/4topics/graph/distro_doc_in_classes.png)
    ![Alt](/src/4topics/graph/distro_doc_in_clusters.png)

2. Classifier comparaison

    Cross-validation accuracy score of 4 classifiers:
    ![Alt](/src/4topics/graph/classifier_comparaison.png)

    Learning-curve:
    ![Alt](/src/4topics/graph/learning_curve_grid.png)

    Test dataset accuracy score if 4 classifiers (number of features choosen: 20):
    ![Alt](/src/4topics/graph/test_scores.png)

## How to play with

1. Copy the document whose topic you want to detect into the file **todetect.txt** under the root directory of project. The text have to be in English, and no preprocessing is required.

2. Run the script **topicDetection.py** under the same directory.

## Issues

1. Due to the lack of labeled documents, the model can only project a document into one of these 4 topics cited above. Even if the document provided has nothing with these topics, it will still be predicted as one of these topics.

## TODO

1. Add more labeled documents about more topics to enlarge the topic diversity of model.

2. Add the fonctionality to predict a document as "UNKNOWN" when the document has a weak similarity to all topics existing in the model.
