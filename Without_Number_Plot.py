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
import matplotlib.pyplot as plt
import pandas as pd
#categories = list(newsgroups_train.target_names)[0:4]
categories =[
 'alt.atheism',
 'comp.graphics',
# 'comp.os.ms-windows.misc',
#'comp.sys.ibm.pc.hardware',
# 'comp.sys.mac.hardware',
# 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
# 'rec.motorcycles',
# 'rec.sport.baseball',
# 'rec.sport.hockey',
 'sci.crypt',
# 'sci.electronics',
# 'sci.med',
# 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
# 'talk.politics.mideast',
# 'talk.politics.misc',
# 'talk.religion.misc'
]
print(categories)


newsgroups_train = fetch_20newsgroups(subset='train',
                                     #remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     #remove=('headers', 'footers', 'quotes'),
                                     categories=categories)

count_vect = CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=0.005,max_df=0.5,token_pattern=u'(?u)\\b[a-zA-Z]{2,20}\\b')#min_df=0.0001, max_df=0.5)#
print(count_vect)
X_train_counts = count_vect.fit_transform(newsgroups_train.data)

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
j = []

for i in range(0,10):
    clf = MultinomialNB(alpha=.01)
    if(i==0):
        j.append(0)
        continue
    clf.fit(X_train_tfidf[0:40*i], newsgroups_train.target[0:40*i])
    
    
    vectors_test2 = count_vect.transform(newsgroups_test.data)
    vectors_test = tfidf_transformer.transform(vectors_test2)
    pred = clf.predict(vectors_test)
    j.append(metrics.precision_score(newsgroups_test.target, pred, average='macro'))
    
    
    
count_vect = CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=0.005,max_df=0.5)#min_df=0.0001, max_df=0.5)#
print(count_vect)
X_train_counts = count_vect.fit_transform(newsgroups_train.data)

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
k = []

for i in range(0,10):
    clf = MultinomialNB(alpha=.01)
    if(i==0):
        k.append(0)
        continue
    clf.fit(X_train_tfidf[0:40*i], newsgroups_train.target[0:40*i])
    
    
    vectors_test2 = count_vect.transform(newsgroups_test.data)
    vectors_test = tfidf_transformer.transform(vectors_test2)
    pred = clf.predict(vectors_test)
    k.append(metrics.precision_score(newsgroups_test.target, pred, average='macro'))    

df4 = pd.DataFrame({"key": i, "value":j})
df3 = pd.DataFrame({"key": i, "value":k})
#df4 = pd.DataFrame.from_dict(df3,orient='index').T
#df4.plot(kind='barh', rot=0,ylim=1)
#df3.join(df4,how='outer').plot()
plt.plot(df3,lw=2.5,label="with Num",color='blue')
plt.plot(df4,lw=2.5,label="without Num",color='red')

plt.ylim(0, 1.0)
plt.show()