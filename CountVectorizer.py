from sklearn.feature_extraction.text import CountVectorizer

content = ("he is a boy. 123 22","She is a girl. 123 23")
print(content)
count_vect = CountVectorizer()
#print(count_vect)
counts = count_vect.fit_transform(content)
print(count_vect.get_feature_names())
print(counts)

count_vect = CountVectorizer(ngram_range=(1,2))
#print(count_vect)
counts = count_vect.fit_transform(content)
print(count_vect.get_feature_names())
print(counts)

count_vect = CountVectorizer(stop_words="english",ngram_range=(1,2))
#print(count_vect)
counts = count_vect.fit_transform(content)
print(count_vect.get_feature_names())
print(counts)

count_vect = CountVectorizer(stop_words="english",ngram_range=(1,2),token_pattern=u'(?u)\\b[a-zA-Z]{2,10}\\b')
#print(count_vect)
counts = count_vect.fit_transform(content)
print(count_vect.get_feature_names())
print(counts)