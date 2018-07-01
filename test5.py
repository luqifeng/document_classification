from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
import re
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk,re
#从gensim.models里导入word2vec
from gensim.models import word2vec
import numpy as np
import matplotlib.pyplot as plt

model = word2vec.Word2Vec.load(u"news.model")

print (model.most_similar('memory'))

X = model[model.wv.vocab]


tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
print(X_tsne[0])
'''
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()