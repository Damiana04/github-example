#!/usr/bin/env python
# coding: utf-8

# ## Non Linear Regression 
# 
# #### 𝑦=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑 
# #### 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)

# In[4]:


# Import packages
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ###### Just recall a linear regression for understanding the difference with a non-linear
# It models a linear relation between a dependent variable y and independent variable x. 

# #### LINEAR REGRESSION y = 2*(x) + 3

# In[5]:


# linear regression: simple equation, of degree 1, for example y =  2𝑥  + 3
x = np.arange(-5.0, 5.0, 0.1)

# Adjusting the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise

# Plt.figure (figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# #### CUBIC y = 1*(x**3) + 1*(x**2) + 1*x + 3

# In[6]:


# Create a CUBIC function's graph
x = np.arange(-5.0, 5.0, 0.1)

# Adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# ### Quadratic 𝑌=𝑋2

# In[7]:


# Create a QUADRATIC function's graph
x = np.arange(-5.0, 5.0, 0.1)

# Adjust the slope and intercept to verify the changes in the graph

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# ### Exponential  Y= np.exp(X)
# 
# 𝑌=𝑎+𝑏𝑐𝑋

# In[8]:


# Create a ESPONENTIAL function's graph
X = np.arange(-5.0, 5.0, 0.1)

# Adjust the slope and intercept to verify the changes in the graph

Y= np.exp(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# ### Logarithmic                                                           Y = np.log(X)
# 
# 𝑦=log(𝑋)

# In[9]:


# Create a LOGARITHMIC function's graph
X = np.arange(-5.0, 5.0, 0.1)

# Adjust the slope and intercept to verify the changes in the graph
Y = np.log(X)

# Plotting
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# ### Sigmoidal/Logistic Y = 1-4/(1+np.power(3, X-2))
# 
# $$ Y = a + \frac{b}{1+ c^{(X-d)}}$$

# In[10]:


# Create a Sigmoidal/Logistic function's graph 
X = np.arange(-5.0, 5.0, 0.1)

# Adjust the slope and intercept to verify the changes in the graph
Y = 1-4/(1+np.power(3, X-2))

# Plotting
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# ## Non-Linear Regression example

# In[11]:


# Import libraries
import numpy as np
import pandas as pd

# Load the data 
df = pd.read_csv("Downloads/china_gdp.csv")
df.head(10)


# In[12]:


# Plotting
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'go')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# ### Trying with a Sigmoidal/Logistic 

# ##### Choosing the model the LOGISTIC function could be a good approximation

# In[13]:


# Create the Esponential function
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

# Plotting
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


# The formula for the logistic function is the following:
# 
# $$ \hat{Y} = \frac1{1+e^{\beta_1(X-\beta_2)}}$$
# 
# $\beta_1$: Controls the curve's steepness,
# 
# $\beta_2$: Slides the curve on the x-axis.

# #### Building The Model

# In[14]:


# Build the regression model and initialize its parameters
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y


# In[1]:


# Try a sample sigmoid line that might fit with the data
beta_1 = 0.10
beta_2 = 1990.0

# Logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

# Plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')


# In[16]:


# Trying to find the best parameters for the model. First normalize x and y

# Normalize the data (Uniform the values with different range)
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)


# ##### Use curve_fit which uses non-linear least squares to fit our sigmoid function, to data. Optimal values for the parameters so that the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.

# In[17]:


# Import packages for curve_fit
from scipy.optimize import curve_fit

# POPT are the optimized parameters
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# Print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


# In[18]:


# Plot the resulting regression model 
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# #### Calculating the ACCURACY of the model

# In[20]:


# SPLIT the data for TRAINING & TESTING
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# FIT/BUILD the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# PREDICT using test set
y_hat = sigmoid(test_x, *popt)

# Evaluation of ACCURACY by MAE (Mean absolute error), MSE (Mean Squared Error/Residual sum of squares) & R2-SCORE
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )


# In[ ]:




