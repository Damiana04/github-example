#!/usr/bin/env python
# coding: utf-8

# ## Simple Linear Regression

# In[22]:


# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


# load data
csv_path = 'FuelConsumptionCo2.csv'
df = pd.read_csv('downloads/FuelConsumptionCo2.csv')

df.head()


# In[9]:


# Checking statistical values  
df.describe()


# In[10]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# In[12]:


# Plot each features into histograms
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# ##### Plot each  features vs the Emission, to see how linear is their relation:

# In[14]:


# Relationship between FUELCONSUMPTION_COMB & CO2EMISSIONS
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[20]:


# Correlation between FUELCONSUMPTION_COMB & CO2EMISSIONS
cdf[["FUELCONSUMPTION_COMB", "CO2EMISSIONS"]].corr()


# In[15]:


# Relationship between CYLINDERS & CO2EMISSIONS
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()


# In[21]:


# Correlation between CYLINDERS & CO2EMISSIONS
cdf[["CYLINDERS", "CO2EMISSIONS"]].corr()


# In[16]:


# Relationship between ENGINESIZE & CO2EMISSIONS
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("Emission")
plt.show()


# In[19]:


# Correlation between ENGINESIZE & CO2EMISSIONS
cdf[["ENGINESIZE", "CO2EMISSIONS"]].corr()


# ### Creating TRAIN and TEST dataset

# In[24]:


# Spliting the dataset in 80% for training, and the 20% for testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# ## Simple Regression Model

# ### Train data distribution

# In[28]:


# Training
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='green')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# ### Model the data distribution

# In[29]:


# Modeling 
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

# Finding the COEFFICIENT & INTERCEPT
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_) 


# ### Plot outputs

# In[32]:


# Plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='purple')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-g')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# ### Evaluation 
# Using MSE as model evaluation metrics
# 
# ###### Mean absolute error: 
# It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
# 
# ###### Mean Squared Error (MSE): 
# Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
# 
# ###### Root Mean Squared Error (RMSE): 
# This is the square root of the Mean Square Error.
# 
# ###### R-squared 
# is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

# In[33]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[ ]:




