#!/usr/bin/env python
# coding: utf-8

# In[56]:


# Random Forest Classification
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[57]:


Sales=pd.read_csv('C:/Users/prate/Downloads/Assignment/Random Forests/Company_Data.csv')


# In[58]:


Sales.head()


# In[59]:


Sales.Sales.mean()


# In[66]:


Sales1=pd.Series(Sales['Sales'])
s=[]
for i in Sales1:
    if i>7.5:
        s.append('good')
    
    else:
        s.append('bad')

print(s)


# In[67]:


Sales2=pd.DataFrame(s)
Sales3=pd.concat([Sales2,Sales],axis=1)
Sales3=Sales3.rename({0:'o/p'},axis=1)
Sales3


# In[69]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

Sales3.iloc[:, 7] = labelencoder.fit_transform(Sales3.iloc[:,7])
Sales3.iloc[:,-2]=labelencoder.fit_transform(Sales3.iloc[:,-2])
Sales3.iloc[:,-1]=labelencoder.fit_transform(Sales3.iloc[:,-1])
Sales3.iloc[:,0]=labelencoder.fit_transform(Sales3.iloc[:,0])


# In[70]:


Sales3.head()


# In[71]:


arrey=Sales3.values


# In[72]:


X=arrey[:,2:10]
Y=arrey[:,0]


# In[76]:


num_trees = 100
max_features = 5
kfold = KFold(n_splits=10,shuffle=True, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[ ]:




