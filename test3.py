from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk,re
#从gensim.models里导入word2vec
from gensim.models import word2vec

news = fetch_20newsgroups(subset='all')
X,y=news.data,news.target
 
#定义一个函数名为news_to_sentences将新闻中的句子逐一剥离出来，并返回一个句子的列表 
def  news_to_sentences(news):
     news_text = BeautifulSoup(news,"html5lib").get_text()
     #print(news_text)
     tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
     raw_sentences = tokenizer.tokenize(news_text)
     #print(raw_sentences)
     sentences=[]
     for sent in raw_sentences:
           sentences.append(re.sub('[^a-zA-Z]',' ',sent.lower().strip()).split())
           #print(re.sub('[^a-zA-Z]',' ',sent.lower().strip()).split())
     return sentences
 
sentences=[]

for x in X:
       sentences += news_to_sentences(x)
       
#print(sentences);
 

 
#配置词向量的维度
num_features = 300
#保证被考虑的词汇的频度
min_word_count = 20
#设定并行化训练使用CPU计算核心的数量，多核可用
num_workers = 4
#定义训练词向量的上下文窗口大小
context = 5
downsampling = 1e-3
 

model = word2vec.Word2Vec(sentences,workers = num_workers,\
                          size = num_features,min_count=min_word_count,\
                          window = context,sample = downsampling)
model.init_sims(replace=True)

model.save(u"news_test.model")

print(model.most_similar('hello'))
 
print(model.most_similar('email'))
print('end')
