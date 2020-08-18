#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# ## Linear Regression and Multiple Linear Regression

# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


# Create the linear regression object

lm = LinearRegression()
lm


# In[7]:


# Define the predictor variable x & the target variable y

X = df[['highway-mpg']]
Y = df['price']


# In[8]:


# FIT the linear model
lm.fit(X,Y)


# In[9]:


# Obtain the PREDICTION
Yhat = lm.predict(X)
Yhat[0:5]


# In[10]:


# Finding the INTERCEPT
lm.intercept_


# In[11]:


# Finding the SLOPE
lm.coef_


# In[12]:


# Relationship between Price & Highway-mpg is given by the formula:
#  y = b0(INTERCEPT) + (in this case - because SLOPE is NEGATIVE) b1(SLOPE) * x(Highway-mpg)


Price = 38423.305858157386-821.73337832 * X


# ### Model Evaluation Linear Regression: 
# ##### Visualization by Regression Plot

# In[53]:


# Import the visualization package: seaborn
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


# Set up the figure size
width = 10
height = 8
plt.figure(figsize=(width, height))

# Call the regression plot down  
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# In[58]:


# Linear Regression Visualizzation between "peak-rpm" & "price"

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# In[59]:


# Correlation between "peak-rpm" & "price"
df[["peak-rpm","highway-mpg","price"]].corr()


# In[13]:


# Linear Regression between 'engine-size' or 'price'

# Create the linear regression object
lm1 = LinearRegression()
lm1


# In[14]:


# Define the predictor variable x & the target variable y. FIT the linear model
lm1.fit(df[['engine-size']], df[['price']])
lm1


# In[15]:


# Intercept
lm1.intercept_


# In[16]:


# Slop 
lm1.coef_


# In[18]:


Yhat = lm1.predict(X)
Yhat[0:5]


# ##  Multiple Linear Regression

# In[19]:


# Extract the 4 predicator variables & store them in a variable Z

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]


# In[21]:


# Train the model
lm.fit(Z, df["price"])


# In[29]:


# Find the coefficient
lm.coef_


# In[30]:


# Find the intercept
lm.intercept_


# In[31]:


# Obtain a prediction
Yhat = lm.predict(Z)
Yhat[0:5]


# ## Model Evaluation Multiple Linear Regression:
# ### Visualization by Distribution Plot

# In[60]:


plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# In[ ]:





# In[46]:


# Create a Linear Regression Object & Train the model

lm2 = LinearRegression()

lm2.fit(df[['normalized-losses' , 'highway-mpg']], df["price"])
lm2


# In[51]:


# Find the coefficient
lm2.coef_


# In[50]:


# Find the intercept
lm2.intercept_


# #### Model Evaluation using Visualization

# In[52]:


# Import the visualization package: seaborn
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


# REGRESSION PLOT: LINEAR REGRESSION

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# ## Polynomial Regression and Pipelines

# In[61]:


# Plot the data

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[62]:


# Create variables
x = df['highway-mpg']
y = df["price"]


# In[66]:


# Calculating a polynomial of the 3rd order (cubic) 
f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p)


# In[67]:


# Plot the function
PlotPolly(p, x, y, 'highway-mpg')


# In[68]:


np.polyfit(x, y, 3)


#         Conclusion: this polynomial model performs better than the linear model.

# In[70]:


# Create 11 order polynomial model with the same variables x ('highway-mpg') and y ('price')
f2 = np.polyfit(x,y,11)
p2 = np.poly1d(f)
print(p2)


# In[72]:


# Plot the function
PlotPolly(p2, x, y, 'Highway MPG')


# In[73]:


# Perform a polynomial transform on multiple features
from sklearn.preprocessing import PolynomialFeatures


# In[74]:


# Create a PolynomialFeatures object of degree 2
pr=PolynomialFeatures(degree=2)
pr


# In[75]:


Z_pr=pr.fit_transform(Z)


# In[76]:


# The original data is of 201 samples and 4 features
Z.shape


# In[77]:


# After the transformation, there 201 samples and 15 features
Z_pr.shape


# ### Pipeline
# ##### Data Pipelines simplify the steps of processing the data.

# In[78]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[79]:


# Creating a list of tuples including the name of the model or estimator and its corresponding constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[81]:


# Input the list as an argument to the pipeline constructor
pipe = Pipeline(Input)
pipe


# In[82]:


# Normalize the data, perform a transform and fit the model simultaneously
pipe.fit(Z,y)


# In[83]:


# Create a prediction
ypipe=pipe.predict(Z)
ypipe[0:4]


# ##### Create a pipeline that Standardizes the data, then perform prediction using a linear regression model using the features Z and targets y

# In[84]:


Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]


# ### Measures for In-Sample Evaluation
# 
# #### R-squared: coefficient of determination, is a measure to indicate how close the data is to the fitted regression line. R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.
# 
# #### Mean Squared Error (MSE): measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).

# ### Simple Linear Regression

# #####    Calcolate the R^2

# In[86]:


# First train highway_mpg_fit
lm.fit(X,Y)

# After find R^2 by lm.score

print('The R-square is: ', lm.score(X, Y))


# ####   Calculate the MSE

# In[87]:


# First find the Yhat
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


# In[88]:


# Import the function mean_squared_error from the module metrics
from sklearn.metrics import mean_squared_error


# In[89]:


# Find the MSE 
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# ###  Multiple Linear Regression

# #####    Calculate the R^2

# In[93]:


# Fit the model 
lm.fit(Z, df['price'])

# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))


# #### Calculate the MSE

# In[100]:


# First find the Yhat making a prediction
Y_predict_multifit = lm.predict(Z)
Y_predict_multifit[0:4]


# In[101]:


# Find the MSE: compare the predicted results with the actual results
print('The mean square error of price and predicted value using multifit is: ',       mean_squared_error(df['price'], Y_predict_multifit))


# ### Polynomial Fit

# #####    Calculate the R^2

# In[102]:


# Import the function r2_score from the module metrics 
from sklearn.metrics import r2_score


# In[104]:


# Apply the function to get the value of R^2
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)


# #### Calculate the MSE

# In[105]:


mean_squared_error(df['price'], p(x))


# ### Prediction and Decision Making

# #### Prediction

# In[118]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[119]:


# Create a new input, generating a sequence of values from 0 to 100, incrementing the sequence one step at the time
new_input=np.arange(1, 100, 1).reshape(-1, 1)


# In[120]:


# Fit the model
lm.fit(X, Y)
lm


# In[121]:


# Produce a prediction 
yhat = lm.predict(new_input)
yhat[0:5]


# In[117]:


plt.plot(new_input, yhat)
plt.show()


# In[ ]:




