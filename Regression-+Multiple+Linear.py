
# coding: utf-8

# In[1]:

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# In[2]:

df = pd.read_excel("ASAP Data2.xlsx")


# In[3]:

#checking the number of rows and columns
df.shape


# In[4]:

list(df.columns.values)


# In[5]:

#Statistical details of the dataset
df.head(10)
#df.describe


# In[6]:

#identify NaNs in the dataset
df.isnull().any()


# In[7]:

#remove NaNs in the dataset if any
df = df.fillna(method='ffill')


# In[8]:

#define the parameters
X = df[['Age', 'Gender', 'Income', 'Driving experience_years', 'Driving frequency', 'Dist driven per week', 'Driv. freq. under inclem. weather', 'Driv. freq. under foggy weather', 'Difficulty driving under fog', 'Health rating',]].values
y = df['Average travel speed (mph)'].values


# In[9]:

#set the test data as 40% and train with 60%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# In[10]:

#train the model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[11]:

#see the optimal coefficients for all variables
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
#coeff_df


# In[12]:

#predict the test data
y_pred = regressor.predict(X_test)


# In[13]:

#check the difference between the actual value and predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 


# In[ ]:

#plot actual vs predicted
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

