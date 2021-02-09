#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf


# # DATA IMPORT

# In[6]:


df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[13]:


df
df.head()


# In[11]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[12]:


df.corr()


# # plot data

# In[29]:


#Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='1')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# 
# from graph we can see that there is positive linear relationship between X & Y
# 

# # preparing Data

# defining the dependent and independent variables

# In[26]:


# Using sklearn 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[27]:


# to divide input and output varibales.
X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# # Training the model

# In[30]:


# Splitting the data into test and training set using train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size= 0.2)


# In[31]:



X_train.shape,Y_train.shape


# In[32]:



X_test.shape,Y_test.shape


# # To fit the model
# 

# In[34]:


model= linear_model.LinearRegression()
model.fit(X_train,Y_train)


# In[37]:



#Plotting the regression line
line = model.coef_*X+model.intercept_
#Plotting for test data
plt.scatter(X,Y)
plt.plot(X,line);
plt.show()


# In[38]:


# Predicting the scores for test set
y_pred= model.predict(X_test)
y_pred


# In[39]:


# Comparing actual versus predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df


# In[40]:



print('Coefficient:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean Squared Error (MSE):%.2f'% mean_squared_error(Y_test,y_pred))
print('Coefficient of Determination (R^2): ', r2_score(Y_test,y_pred))


# In[46]:


#Let's predict the score for 9.25 hpurs
print('Score of student who studied for 9.25 hours a dat', Regressor.predict([[9.25]]))


# In[ ]:




