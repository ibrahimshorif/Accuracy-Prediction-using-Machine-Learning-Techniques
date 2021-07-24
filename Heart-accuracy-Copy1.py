#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# In[2]:


#Read the training & test data
hdf = pd.read_csv(r'C:\Users\SWARNA\Desktop\heart.csv')


# In[4]:


hdf.head()


# In[5]:


#create dependent &independent variable vectors
x=hdf.iloc[:,:-1].values
y=hdf.iloc[:,-1].values
print(x)
print(y)


# In[6]:


#Check for any null values
hdf.isnull().sum()


# In[7]:


from sklearn.preprocessing import StandardScaler
  
# Initialise the Scaler
scaler = StandardScaler()
  


# In[8]:


# To scale data
scaler.fit(hdf)


# In[17]:


a = hdf.iloc[:, 2:12].values
print ("\nOriginal data values : \n",  a)
  


# In[19]:


from sklearn import preprocessing
  
#""" MIN MAX SCALER """
  
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
  
# Scaled feature
a_after_min_max_scaler = min_max_scaler.fit_transform(a)
  
print ("\nAfter min max Scaling : \n", a_after_min_max_scaler)
  


# In[20]:


#""" Standardisation """
  
Standardisation = preprocessing.StandardScaler()
  
# Scaled feature
a_after_Standardisation = Standardisation.fit_transform(a)
  
print ("\nAfter Standardisation : \n", a_after_Standardisation)


# In[21]:


# Importing modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)


# In[27]:


#2) Logistic Regression
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(x_train, y_train)
#Predict Output
log_predicted= logreg.predict(x_test)

logreg_score = round(logreg.score(x_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(x_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
print('Classification Report: \n', classification_report(y_test,log_predicted))


# In[26]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
#Predict Output
gauss_predicted = gaussian.predict(x_test)

gauss_score = round(gaussian.score(x_train, y_train) * 100, 2)
gauss_score_test = round(gaussian.score(x_test, y_test) * 100, 2)
print('Gaussian Score: \n', gauss_score)
print('Gaussian Test Score: \n', gauss_score_test)
print('Accuracy: \n', accuracy_score(y_test, gauss_predicted))
print(confusion_matrix(y_test,gauss_predicted))
print(classification_report(y_test,gauss_predicted))


# In[28]:


#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(x_test)


# In[29]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[30]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[31]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:




