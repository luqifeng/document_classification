from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
categories = list(newsgroups_train.target_names)[0:4]

print(categories)


newsgroups_train = fetch_20newsgroups(subset='train',
                                     #remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     #remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
print(newsgroups_train.data[:10])
count_vect = CountVectorizer(stop_words='english',ngram_range=(1,2),token_pattern=u'(?u)\\b[a-zA-Z]{2,20}\\b',min_df=0.005,max_df=0.5)#min_df=0.0001, max_df=0.5)#
print(count_vect)
X_train_counts = count_vect.fit_transform(newsgroups_train.data)
print(count_vect.get_feature_names()[:100])
#print(X_train_counts.toarray()[0])

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB(alpha=.01)
#clf = svm.LinearSVC()
clf.fit(X_train_tfidf, newsgroups_train.target)

#pred_t = clf.predict(X_train_tfidf)
#print(metrics.precision_score(newsgroups_train.target, pred_t, average='macro'))

vectors_test2 = count_vect.transform(newsgroups_test.data)
vectors_test = tfidf_transformer.transform(vectors_test2)
pred = clf.predict(vectors_test)
print(metrics.precision_score(newsgroups_test.target, pred, average='macro'))