#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[4]:


import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[5]:


df=pd.read_csv('https://raw.githubusercontent.com/chaitanyabaranwal/ParkinsonAnalysis/master/parkinsons.csv')
df.head()


# In[6]:


features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[7]:


print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[8]:


scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[12]:


model=XGBClassifier()
model.fit(x_train,y_train)


# In[13]:


y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




