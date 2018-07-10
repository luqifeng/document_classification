from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.feature_selection import SelectKBest,chi2
newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint

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
#categories = list(newsgroups_train.target_names)

print(categories)


newsgroups_train = fetch_20newsgroups(subset='train',
                                     #remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     #remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
print(newsgroups_train.data[:10])
count_vect = CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=0.005,max_df=0.5,token_pattern=u'(?u)\\b[a-zA-Z]{2,20}\\b')# min_df=0.0001, max_df=0.5)#
print(count_vect)
X_train_counts = count_vect.fit_transform(newsgroups_train.data)
print(count_vect.get_feature_names()[:100])
#print(X_train_counts.toarray()[0])

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

feature_names = count_vect.get_feature_names()
ch2 = SelectKBest(chi2, k=1500)
X_train = ch2.fit_transform(X_train_tfidf, newsgroups_train.target)

selected_feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]


#clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.3,max_depth=3, random_state=0)
#clf = MultinomialNB(alpha=0.1)
#clf = svm.LinearSVC(max_iter = 2000)
clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")


clf.fit(X_train, newsgroups_train.target)

pred_t = clf.predict(X_train)
print(metrics.precision_score(newsgroups_train.target, pred_t, average='macro'))

vectors_test2 = count_vect.transform(newsgroups_test.data)
vectors_test = tfidf_transformer.transform(vectors_test2)
X_test = ch2.transform(vectors_test)
pred = clf.predict(X_test)
print(metrics.precision_score(newsgroups_test.target, pred, average='macro'))

