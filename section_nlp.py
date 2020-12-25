#!/usr/bin/env python
# coding: utf-8

# In[54]:


import nltk
import random
from nltk.corpus import brown
brown.categories()


# In[55]:


documents = []
for category in brown.categories():
    for fileid in brown.fileids(category):
        documents.append([brown.words(fileid), category])
print(documents[100])
print(documents[300])
print(documents[200])


# In[56]:


len(documents)


# In[57]:


from nltk.corpus import stopwords
stop = stopwords.words("english")


# In[58]:


words = []
for word in brown.words():
    if word not in stop:
        words.append(word.lower())
len(words)


# In[75]:


words_fD = nltk.FreqDist(words)
words_fD.most_common(10000)


# In[76]:


words_fD['new']


# In[88]:


word_features = list(words_fD.keys())[:10000]


# In[90]:


featuresets = []
for (word, category) in documents:
    words =list(word)
    features ={}
    for w in word_features:
        features[w] = w in words
    featuresets.append([features, category])
len(featuresets)


# In[91]:


featuresets[100]


# In[92]:


len(featuresets)


# In[93]:


len(featuresets[0][0])


# In[97]:


training_set =featuresets[:450]
testing_set = featuresets[50:]


# In[99]:


classf = nltk.NaiveBayesClassifier.train(training_set)
print('accurcy for model :',nltk.classify.accuracy(classf, testing_set))

