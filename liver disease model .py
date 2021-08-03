#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("E:\level 4 term 1\CSE 441\patient.csv")


# In[3]:


df.head()


# In[4]:


#dependent and indepandent variable
#independent 
x= df.iloc[:,:-1].values
#dependent 
y= df.iloc[:,-1].values


# In[5]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[6]:


x_train
x_test
y_train
y_test


# # Encoding 

# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[10]:


df['Gender'] = le.fit_transform(df['Gender'])


# In[11]:


df.head()


# # Missing values Hundle
# 

# In[13]:


#count missing values
print(df.isnull().sum())


# In[14]:


#drop missing value record
df.dropna(inplace=True)


# In[15]:


df.head()


# In[16]:


df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())


# In[17]:


#count missing values
print(df.isnull().sum())


# In[18]:


#dependent and indepandent variable
#independent 
x= df.iloc[:,:-1].values
#dependent 
y= df.iloc[:,-1].values


# In[21]:


print(x)
print(y)


# In[23]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[25]:


x_test.shape


# In[26]:


y_test.shape


# # Feature slection 

# In[27]:


#unarative
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

FIT_FEATURES = SelectKBest(score_func=f_classif)

FIT_FEATURES.fit(x,y)


# In[28]:


pd.DataFrame(FIT_FEATURES.scores_)


# In[29]:


SCORE_COL = pd.DataFrame(FIT_FEATURES.scores_,columns =['Score'])
SCORE_COL


# In[30]:


SCORE_COL.nlargest(8,'Score')


# In[31]:


df = df.drop(['Gender','Albumin'],axis=1)


# In[32]:


df.head()


# In[33]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[34]:


x_test


# In[35]:


y_test


# # Feature scaling 

# In[36]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train[:,1:] = scaler.fit_transform(x_train[:,1:])
x_test[:,1:] = scaler.fit_transform(x_test[:,1:])
 


# In[37]:


x_train.shape


# In[39]:


y_train.shape


# # LogisticRegression

# In[40]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[41]:


model.predict(x_test)


# In[42]:


model.score(x_test,y_test)*100


# In[43]:


pred_log = model.predict(x_test)


# In[44]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[45]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # DecisionTree

# In[46]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[47]:


from sklearn.tree import DecisionTreeClassifier 
model01= DecisionTreeClassifier()
model01.fit(x_train,y_train)


# In[48]:


model01.predict(x_test)


# In[49]:


model01.score(x_test,y_test)*100


# In[50]:


pred_log = model01.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[51]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# 
# # KNN

# In[52]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[53]:


from sklearn.neighbors import KNeighborsClassifier
model02= KNeighborsClassifier()
model02.fit(x_train,y_train)


# In[54]:


model02.predict(x_test)


# In[55]:


model02.score(x_test,y_test)*100


# In[56]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[57]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # Random Forest

# In[58]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=20, random_state = 10)


# In[60]:


from sklearn.ensemble import RandomForestClassifier
model02= RandomForestClassifier()
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


# # Naive Bayse

# In[65]:


from sklearn.naive_bayes import GaussianNB
model02= GaussianNB()
model02.fit(x_train,y_train)


# In[66]:


model02.predict(x_test)


# In[67]:


model02.score(x_test,y_test)*100


# In[68]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[69]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# # SVM

# In[70]:


from sklearn.svm import SVC
model02= SVC()
model02.fit(x_train,y_train)


# In[71]:


model02.predict(x_test)


# In[72]:


model02.score(x_test,y_test)*100


# In[73]:


pred_log = model02.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_log)


# In[74]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_log))


# In[ ]:




