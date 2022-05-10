#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[4]:


pima=data('Pima.tr')


# In[10]:


pima.plot(kind='scatter',x='skin',y='bmi')
plt.show()


# In[6]:


#Test train split data for supervised learning
X_train,X_test,y_train,y_test=train_test_split(pima.skin,pima.bmi)


# In[11]:


#test train split visualisation
plt.scatter(X_train,y_train, label='Training Data',color='r',alpha=0.7)
plt.scatter(X_test, y_test, label='Testing Data', color='g', alpha=0.7)
plt.legend()
plt.title('Test Train Split')
plt.show()


# In[13]:


#create linear model and train it
LR= LinearRegression()
LR.fit(X_train.values.reshape(-1,1),y_train.values) #(-1,1) gives all matrix into 1D array


# In[17]:


prediction=LR.predict(X_test.values.reshape(-1,1))
#plotprediction
plt.plot(X_test, prediction, label='Linear Regression', color='b')
plt.scatter(X_test,y_test, label='Actual Test Data', color='g', alpha=0.8)
plt.legend()
plt.show()


# In[24]:


#predict bmi of person with skin fold 50
LR.predict(np.array([[40]]))[0]


# In[25]:


#Score this model
LR.score(X_test.values.reshape(-1,1),y_test.values)


# In[ ]:


("")

