

from __future__ import print_function

from pprint import pprint
from time import time
import logging

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
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
# Load some categories from the training set
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

#categories = categories[0:2]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

data = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier

pipeline = Pipeline([
    ('vect', CountVectorizer(analyzer='word',stop_words='english')),
    #('reducer', SelectKBest(k=5000)),
    ('tfidf', TfidfTransformer()),
#    ('clf', SGDClassifier()),

#    ('svm', NearestCentroid()),
#    ('MLP',MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(100,20), random_state=1,max_iter=30,verbose=10,learning_rate_init=.1)),        
    ('svm', svm.LinearSVC()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__min_df':(0.01,),
    'vect__max_df': (0.5,),# 0.75, 1.0),
    #'vect__max_features': (20000,None),#(None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 2),),  # unigrams or bigrams
    'tfidf__use_idf': (True,),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.001, 0.0001),
    #'clf__penalty': ('l2','l1' ),
    #'clf__n_iter': (20,),
    #'svm__decision_function_shape': ('ovr',),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("----------------------TEST------------------------")
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories=categories)
    y_pred = grid_search.predict(newsgroups_test.data)
    print(classification_report(y_true=newsgroups_test.target, y_pred=y_pred))    
