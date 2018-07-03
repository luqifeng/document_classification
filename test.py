#from pyecharts import Bar
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
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
categories = list(newsgroups_train.target_names)
print(newsgroups_train.filenames.shape)
#print(newsgroups_train.filenames[:10])
print(newsgroups_train.target.shape)
#print(newsgroups_train.target[:10])
#news = fetch_20newsgroups(subset='all')  


#cats = ['alt.atheism', 'sci.space']
#newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
#print(list(newsgroups_train.target_names))
#print(newsgroups_train.target.shape)
#print(newsgroups_train.target[:10])
#ÊýÁ¿Í¼
from sklearn.feature_extraction.text import TfidfVectorizer
#categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
#newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors)
print(vectors.shape)
'''



print(vectors.nnz / float(vectors.shape[0]))



newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-100:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

show_top10(clf, vectorizer, newsgroups_train.target_names)
#pprint(vectorizer.get_feature_names())


vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
print(metrics.f1_score(pred, newsgroups_test.target, average='macro'))
'''
newsgroups_train = fetch_20newsgroups(subset='train',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories=categories)
vectors = vectorizer.fit_transform(newsgroups_train.data)
#clf = MultinomialNB(alpha=.01)
clf = SGDClassifier()
clf.fit(vectors, newsgroups_train.target)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))