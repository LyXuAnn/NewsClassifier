# -*- coding: <utf-8> -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import numpy as np;
from sklearn.naive_bayes import MultinomialNB
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
# term frequency
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# TFIDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# bayes to predict
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

# estimate precision
# twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = twenty_test.data
X_test_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_test_tfidf)
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print("accuracy\t"+str(np.mean(predicted == twenty_test.target)))

