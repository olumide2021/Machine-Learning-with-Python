
# coding: utf-8

# In[77]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# In[17]:

df = pd.read_excel("ASAP Data2.xlsx")


# In[18]:

#df = pd.read_csv("ASAP Data.csv")


# In[21]:

#checking the number of rows and columns
df.shape


# In[25]:

list(df.columns.values)


# In[31]:

#Statistical details of the dataset
df
#df.describe


# In[38]:

#find any relationship between the data
df.plot(x='Age', y='Crash frequency (collisions )', style='o')
plt.title('Age vs Crash frequency (collisions )')  
plt.xlabel('Age')  
plt.ylabel('Crash frequency (collisions )')  
plt.show()


# In[47]:

#define x and y and reshape the dataset
X = df['Age'].values.reshape(-1,1)
y = df['Crash frequency (collisions )'].values.reshape(-1,1)


# In[68]:

#set the test data as 20% and train with 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[69]:

#define the linear regression as regressor
regressor = LinearRegression()  
#training the algorithm
regressor.fit(X_train, y_train) 


# In[70]:

#retrieve the intercept:
print(regressor.intercept_)


# In[71]:

#retrieve the slope:
print(regressor.coef_)


# In[72]:

#use test data and see how accurately the algorithm predicts the percentage score in step 59
y_pred = regressor.predict(X_test)


# In[73]:

#compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[74]:

#visualize comparison result as a bar graph
df1 = df.head(15)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[75]:

#plot a straight line with the test data
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[76]:

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:



