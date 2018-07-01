from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups

#导入数据
newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
pprint(list(newsgroups_train.target_names))
print(newsgroups_train.filenames.shape)
print(newsgroups_train.target.shape)


#计算文件数量
number = []
kkk = list(newsgroups_train.target_names)
for i in kkk:
    print i
    target = fetch_20newsgroups(subset='all', categories=[i])
    number.append(target.filenames.shape[0])
print (number) 

#绘制柱状图
plt.rcdefaults()
fig, ax = plt.subplots()
people = list(newsgroups_train.target_names)
y_pos = np.arange(len(people))
error = np.random.rand(len(people))
ax.barh(y_pos, number, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis() 
ax.set_xlabel('Number')
ax.set_title('Number Of Groups')
plt.show()