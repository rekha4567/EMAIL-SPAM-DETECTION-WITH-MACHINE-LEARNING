#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string


# In[85]:


df = pd.read_csv("email.csv",encoding='ISO-8859-1')
df.head()


# In[86]:


df.shape


# In[87]:


df.columns


# In[88]:


df.drop_duplicates(inplace=True)
print(df.shape)


# In[89]:


print(df.isnull().sum())


# In[73]:


nltk.download("stopwords")


# In[90]:


data.columns


# In[91]:


data.isna().sum()


# In[92]:


data['Spam']=data['v1'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)


# In[93]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.v2,data.Spam,test_size=0.25)


# In[94]:


#CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer


# In[95]:


from sklearn.naive_bayes import MultinomialNB


# In[96]:


from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])


# In[97]:


clf.fit(X_train,y_train)


# In[98]:


emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]


# In[99]:


clf.predict(emails)


# In[100]:


clf.score(X_test,y_test)


# In[ ]:




