from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk,re
#��gensim.models�ﵼ��word2vec
from gensim.models import word2vec

news = fetch_20newsgroups(subset='all')
X,y=news.data,news.target
 
#����һ��������Ϊnews_to_sentences�������еľ�����һ���������������һ�����ӵ��б� 
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
 

 
#���ô�������ά��
num_features = 300
#��֤�����ǵĴʻ��Ƶ��
min_word_count = 20
#�趨���л�ѵ��ʹ��CPU������ĵ���������˿���
num_workers = 4
#����ѵ���������������Ĵ��ڴ�С
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
