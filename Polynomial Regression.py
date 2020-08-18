#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


csv_path = "FuelConsumptionCo2 (1).csv"
df = pd.read_csv("downloads/FuelConsumptionCo2 (1).csv")
df.head()


# In[8]:


# Select some features that we want to use for regression 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# In[78]:


# Plot Emission values with respect to Engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='pink')
plt.xlabel('ENGINESIZE')
plt.xlabel("EMISSIONS")
plt.show()


# #### Creating train and test dataset

# In[80]:


# Training & Splitting the dataset
msk = np.random.rand(len(df)) <0.8
train = cdf[msk]
test = cdf[~msk]


# When the trend of data is not really linear, we need to use Polynomial regression methods. 
# That means that we need to train & test the dataset by a the PloynomialFeatures() function that allows to create a polynomial with specified degree.
# After done this, by fit_transform we transform our data from power of 0 to power of 2 (since we set the degree of our polynomial to 2).

# In[81]:


# Importing packages
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Setting the Training & Testing data for transforming the trend into a Polynomial Regreassion (it was a simple linear regression)
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Creating & Transorming the trend of data in a Polynomial with degree 2
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly


# In[84]:


# Now we can use LinearRegression() function & find Coefficient & Intercept
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


# Coefficient and Intercept , are the parameters of the fit curvy line.

# In[85]:


# Plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='green')
XX = np.arange(0.0, 10.0, 0.1) # array from 0 to 10, incrasing 1 point by 1
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-p' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


# ### Evaluation: MAE, MSE, R2-SCORE

# In[87]:


# Importing packages
from sklearn.metrics import r2_score

# Fitting, Transforming & Predicting
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

# MAE, MSE, R2-SCORE values
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# ##### Using a polynomial regression with the dataset with CUBIC DEGREE. Does it result in better accuracy?

# In[88]:


# Creating & Transorming the trend of data in a Polynomial with degree 3
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
train_x_poly3


# In[89]:


# Use the LinearRegression() function & find Coefficient & Intercept
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)

# Coefficients
print ('Coefficients: ', clf3.coef_)
print("Intercept:", clf3.intercept_)


# In[90]:


# Plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='purple')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX+ clf3.coef_[0][2]* np.power(XX, 2)+ clf3.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r')
plt.xlabel("Engine-size")
plt.xlabel("Co2Emissions")


# In[77]:


# Evaluation by MaE, MsE, r2-score
from sklearn.metrics import r2_score

#Fitting, Transorming & Predicting
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)

# MaE, MsE, r2-score values
print("Mean absolute Error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of square (MsE): %.2f" % np.mean((test_y3_ - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y3_, test_y) )


# In[ ]:





# In[ ]:




