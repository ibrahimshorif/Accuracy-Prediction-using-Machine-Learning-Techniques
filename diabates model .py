#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd


# In[13]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\diabetes.csv")


# In[14]:


df.head()


# In[15]:


#count missing values
print(df.isnull().sum())


# In[16]:


#dependent and indepandent variable
#independent 
x= df.iloc[:,:-1].values
#dependent 
y= df.iloc[:,-1].values


# In[17]:


print(x.shape)
print(y.shape)


# In[18]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[19]:


x_test


# In[20]:


y_test


# In[45]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train[:,1:] = scaler.fit_transform(x_train[:,1:])
x_test[:,1:] = scaler.fit_transform(x_test[:,1:])
 


# In[46]:


x_train


# In[47]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[48]:


model.predict(x_test)


# In[49]:


model.score(x_test,y_test)*100


# In[50]:


pred_log = model.predict(x_test)


# In[51]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[53]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[54]:


from sklearn.tree import DecisionTreeClassifier 
model01= DecisionTreeClassifier()
model01.fit(x_train,y_train)


# In[55]:


model01.predict(x_test)


# In[56]:


model01.score(x_test,y_test)*100


# In[57]:


pred_log = model01.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[58]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[59]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[60]:


from sklearn.neighbors import KNeighborsClassifier
model02= KNeighborsClassifier()
model02.fit(x_train,y_train)


# In[61]:


model02.predict(x_test)


# In[62]:


model02.score(x_test,y_test)*100


# In[63]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[64]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[ ]:





# In[ ]:





# In[ ]:




