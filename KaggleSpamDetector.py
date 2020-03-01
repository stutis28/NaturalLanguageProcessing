#!/usr/bin/env python
# coding: utf-8

# Kaggle Spam Detector

# In[26]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("*****\\personal\spam.csv", encoding = 'ISO-8859-1')


# In[3]:


df.head()


# In[4]:


# drop unneccesary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
df.head()


# In[5]:


# rename columns
df.columns = ['label','data']
df.head()


# In[6]:


# create binary labels
df['b_labels'] = df['label'].map({'ham':0,'spam':1})
Y = df['b_labels'].as_matrix()


# In[7]:


count_vectorizer = CountVectorizer(decode_error = 'ignore')
X = count_vectorizer.fit_transform(df['data'])


# In[19]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size = 0.33)


# In[21]:


model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:",model.score(Xtrain,Ytrain))
print("test score:",model.score(Xtest,Ytest))


# In[27]:


#visualize data
def visualize(label):
    words = ''
    for msg in df[df['label'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width =600,height =400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize('spam')
visualize('ham')


# In[28]:


# to see what we are getting wrong
df['predictions'] = model.predict(X)


# In[29]:


# things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)


# In[30]:


# things that should be not spam
sneaky_nospam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in sneaky_nospam:
    print(msg)


# In[ ]:




