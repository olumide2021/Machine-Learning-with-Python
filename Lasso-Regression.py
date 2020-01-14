
# coding: utf-8

# In[50]:

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[51]:

# difference of lasso and ridge regression is that some of the coefficients can be zero i.e. some of the features are 
# completely neglected
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split


# In[57]:

cancer = load_breast_cancer()
cancer.keys()


# In[56]:

cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df.head(3)


# In[33]:

X = cancer.data
Y = cancer.target


# In[34]:

X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)


# In[35]:

lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)


# In[63]:

"training score:", train_score


# In[64]:

"test score: ", test_score


# In[65]:

"number of features used: ", coeff_used


# In[66]:

lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)


# In[67]:

train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)


# In[68]:

"training score for alpha=0.01:", train_score001 


# In[69]:

"test score for alpha =0.01: ", test_score001


# In[70]:

"number of features used: for alpha =0.01:", coeff_used001


# In[72]:

train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)


# In[22]:

#print "training score for alpha=0.0001:", train_score00001 
#print "test score for alpha =0.0001: ", test_score00001
#print "number of features used: for alpha =0.0001:", coeff_used00001


# In[44]:

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)


# In[45]:

#print "LR training score:", lr_train_score
#print "LR test score: ", lr_test_score


# In[73]:

plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency


# In[74]:

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)


# In[75]:

plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()


# In[ ]:



