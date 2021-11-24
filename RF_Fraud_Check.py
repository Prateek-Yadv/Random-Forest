#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


fraud=pd.read_csv('C:/Users/prate/Downloads/Assignment/Random Forests/Fraud_check.csv')


# In[4]:


fraud.head()


# In[13]:


Fraud=fraud.rename({'Marital.status':'Marital_status','Taxable.Income':'Taxable_Income','City.Population':'City_Population','Work.Experience':'Work_Experience'}
                  ,axis=1)


# In[14]:


Fraud.head()


# In[22]:


Fraud_df=pd.Series(Fraud['Taxable_Income'])
sf=[]
for i in Fraud_df:
    if i<=30000:
        sf.append('risky')
    else:
        sf.append('good')
print(sf)


# In[24]:


Fraud_df=pd.DataFrame(sf)
Fraud_df=pd.concat([Fraud_df,Fraud],axis=1)
Fraud_df=Fraud_df.rename({0:'o/p'},axis=1)
Fraud_df


# In[25]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

Fraud_df.iloc[:,0] = labelencoder.fit_transform(Fraud_df.iloc[:,0])
Fraud_df.iloc[:,1]=labelencoder.fit_transform(Fraud_df.iloc[:,1])
Fraud_df.iloc[:,2]=labelencoder.fit_transform(Fraud_df.iloc[:,2])
Fraud_df.iloc[:,-1]=labelencoder.fit_transform(Fraud_df.iloc[:,-1])


# In[26]:


Fraud_df.head()


# In[27]:


arrey=Fraud_df.values


# In[28]:


X=arrey[:,1:7]
Y=arrey[:,0]


# In[29]:


num_trees = 100
max_features = 4
kfold = KFold(n_splits=10,shuffle=True, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[ ]:




