#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\heart.csv")


# In[4]:


df


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


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[10]:


from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(x_train ,x_test)
robust_scaled_df = pd.DataFrame(robust_scaled_df)
  


# In[11]:


x_train.shape


# In[ ]:




