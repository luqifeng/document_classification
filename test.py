#from pyecharts import Bar

import numpy as np
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
pprint(list(newsgroups_train.target_names))
print(newsgroups_train.filenames.shape)
#print(newsgroups_train.filenames[:10])
print(newsgroups_train.target.shape)
print(newsgroups_train.target[:10])
#news = fetch_20newsgroups(subset='all')  


cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
print(list(newsgroups_train.target_names))
print(newsgroups_train.target.shape)
print(newsgroups_train.target[:10])
#����ͼ
'''
number = []
kkk = list(newsgroups_train.target_names)
for i in kkk:
    print i
    target = fetch_20newsgroups(subset='all', categories=[i])
    number.append(target.filenames.shape[0])
print (number) 

bar = Bar("��������ͼ", "",width=800,height=400)
bar.add("����", list(newsgroups_train.target_names),number,is_convert=True,yaxis_interval=0, xaxis_rotate=30, yaxis_rotate=30,yaxis_label_textsize=9)
#bar.show_config()
bar.render()
'''

from sklearn.feature_extraction.text import TfidfVectorizer
categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

print(vectors.nnz / float(vectors.shape[0]))

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

show_top10(clf, vectorizer, newsgroups_train.target_names)
pprint(vectorizer.get_feature_names())

newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
print(metrics.f1_score(pred, newsgroups_test.target, average='macro'))

newsgroups_train = fetch_20newsgroups(subset='train',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))