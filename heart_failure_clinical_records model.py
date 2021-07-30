#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\heart_failure_clinical_records.csv")


# In[3]:


df.head()


# In[4]:


#count missing values
print(df.isnull().sum())


# In[5]:


#dependent and indepandent variable
#independent 
x= df.iloc[:,:-1].values
#dependent 
y= df.iloc[:,-1].values


# In[6]:


print(x)
print(y)


# In[7]:


#unarative
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

FIT_FEATURES = SelectKBest(score_func=f_classif)

FIT_FEATURES.fit(x,y)


# In[8]:


pd.DataFrame(FIT_FEATURES.scores_)


# In[9]:


SCORE_COL = pd.DataFrame(FIT_FEATURES.scores_,columns =['Score'])
SCORE_COL


# In[10]:


SCORE_COL.nlargest(8,'Score')


# In[11]:


df = df.drop(['diabetes','smoking','time'],axis=1)


# In[12]:


df.head()


# In[13]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[14]:


x_test.shape


# In[16]:


y_test


# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train[:,1:] = scaler.fit_transform(x_train[:,1:])
x_test[:,1:] = scaler.fit_transform(x_test[:,1:])
 


# In[18]:


x_train.shape


# In[19]:


y_train.shape


# In[20]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[21]:


model.predict(x_test)


# In[22]:


model.score(x_test,y_test)*100


# In[23]:


pred_log = model.predict(x_test)


# In[24]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[27]:


#decision_tree
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[28]:


from sklearn.tree import DecisionTreeClassifier 
model01= DecisionTreeClassifier()
model01.fit(x_train,y_train)


# In[29]:


model01.predict(x_test)


# In[30]:


model01.score(x_test,y_test)*100


# In[31]:


pred_log = model01.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[33]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
model02= KNeighborsClassifier()
model02.fit(x_train,y_train)


# In[35]:


model02.predict(x_test)


# In[36]:


model02.score(x_test,y_test)*100


# In[37]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[ ]:




