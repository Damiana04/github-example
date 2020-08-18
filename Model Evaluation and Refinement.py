#!/usr/bin/env python
# coding: utf-8

# ## Model Evaluation and Refinement

# In[1]:


# Import libraries & packages

get_ipython().system(' pip install ipywidgets')

import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from IPython.display import display
from IPython.html import widgets 
from IPython.display import display

from ipywidgets import interact, interactive, fixed, interact_manual

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[2]:


# Import clean data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)
# Save the file
df.to_csv('module_5_auto.csv')


# In[3]:


# Using just NUMERIC data 
df=df._get_numeric_data()
df.head()


# ### Functions for plotting

# In[4]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


# In[57]:


# Define a Polynomial function for plotting later
def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    
    
    # Training & Testing data 
    # lr:  linear regression object 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)



    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    
    # Poly_transform:  polynomial transformation object
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


# ## Training and Testing

# ##### An important step: TESTING the model, SPLITING the data into training and testing data. 

# In[6]:


# PLACING the TARGET price_data in a separate dataframe y
y_data = df['price']


# In[7]:


# DROP price data in x data
x_data=df.drop('price',axis=1)


# In[8]:


# Randomly split the data into training and testing data, using the function train_test_split
# The test_size parameter sets the proportion of data that is split into the testing set. The testing set is set to 15% of the total dataset.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("Number of test sample:", x_test.shape[0])
print("Number of training sample:", x_train.shape[0])


# ###### Use the function "train_test_split" to split up the data set such that 40% of the data samples will be utilized for testing, set the parameter "random_state" equal to zero. The output of the function should be the following: "x_train_1" , "x_test_1", "y_train_1" and "y_test_1".

# In[9]:


x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)
print("Number of test sample:", x_test_1.shape[0])
print("Number of training sample:", x_train_1.shape[0])


# In[10]:


# Now use LinearRegression: import package
from sklearn.linear_model import LinearRegression


# In[11]:


# Create a LinearRegression object
lre = LinearRegression()


# In[12]:


# Fit the model (feature "houspower")
lre.fit(x_train[['horsepower']], y_train)


# In[13]:


# Calculate the R^2 on the TEST data (test sample: 31 = 15%)
lre.score(x_test[['horsepower']], y_test)


# In[14]:


# Calculate the R^2 on the TRAIN data (training sample: 170 = 85%)
lre.score(x_train[['horsepower']], y_train)


# Note: R^2 is much SMALLER using the TEST data.
# ##### INTERPRETATION of R-SQUARE (coefficient of determination) is how well the regression model fits the observed data. 
# ###### Generally, a higher r-squared indicates a better fit for the model.
# ###### For example, an r-squared of 60% reveals that 60% of the data fit the regression model.
# 
# Exactly –1. A perfect downhill (negative) linear relationship
# 
# –0.70. A strong downhill (negative) linear relationship
# 
# –0.50. A moderate downhill (negative) relationship
# 
# –0.30. A weak downhill (negative) linear relationship
# 
# 0. No linear relationship
# 
# +0.30. A weak uphill (positive) linear relationship
# 
# +0.50. A moderate uphill (positive) relationship
# 
# +0.70. A strong uphill (positive) linear relationship
# 
# Exactly +1. A perfect uphill (positive) linear relationship
# 

# ##### Find the R^2 on the test data using 90% of the data for training data

# In[15]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
print("Number of test sample:", x_test.shape[0])
print("Number of training sample:", x_train.shape[0])


# In[16]:


# Fit the model (feature "houspower")
lre.fit(x_train1[['horsepower']], y_train1)


# In[17]:


# Calculate the R^2 on the TEST data (test sample: 21 = 10%)
lre.score(x_test1[['horsepower']], y_test1)


# In[18]:


# Calculate the R^2 on the TRAIN data (training sample: 180 = 90%)
lre.score(x_train1[['horsepower']], y_train1)


# In[19]:


# Plotting training & testing data
plt.plot(x_train, y_train, 'ro', label='Training Data')
plt.plot(x_test, y_test, 'go', label='Test Data')
    


# #### Cross-validation Score

# In[20]:


# Import model_selection from the module cross_val_score
from sklearn.model_selection import cross_val_score


# In[21]:


# Input the OBJECT (feature 'horsepower'); the TARGET data (y_data = 'price'); the PARAMETER CV to 4 (4=number of folds).
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)


# In[22]:


# The default scoring is R^2. Just calling the name of the object, are returned each element in the array, which has the average R^2 value in the fold
Rcross


# In[23]:


# Calculate the AVERAGE and STANDARD DEVIATION of our estimate
print("The mean of the folds are", Rcross.mean())
print("The standard deviation is" , Rcross.std())


# In[24]:


# Alternative to calculate the average
np.mean(Rcross)


# In[25]:


# Alternative to calculate the standard deviation
np.std(Rcross)


# In[26]:


# Taken it from website of scikit-learn
print("Accuracy: %0.2f (+/- %0.2f)" % (Rcross.mean(), Rcross.std() * 2))


# In[27]:


# We can use 'negative squared error' function as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')


# In[28]:


# Calculate the average R^2 using two folds. Find the average R^2 for the second fold utilizing the horsepower as a feature 
Rcross2 = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)


# In[29]:


Rcross2


# In[30]:


np.mean(Rcross2)


# In[31]:


np.std(Rcross2)


# In[32]:


print("Accuracy: %0.2f (+/- %0.2f)" % (Rcross.mean(), Rcross.std() * 2))


# In[33]:


print(Rcross.std() * 2)


# ###### Prediction in Cross Validation

# In[34]:


# Import packages
from sklearn.model_selection import cross_val_predict


# In[35]:


# Predicting 
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
yhat[0:5]


# #### Overfitting, Underfitting and Model Selection

# In[36]:


# Creating Multiple linear regression objects  
lr = LinearRegression()


# In[37]:


# Train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features 
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)


# In[38]:


# Prediction using training data
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]


# In[39]:


# Prediction using testing data
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]


# In[40]:


# Perform & Plot some model evaluation using the training and testing data separately
# First import packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ##### Plot of predicted value using the training data compared to the train data

# In[41]:


# Examine the distribution of the predicted values of the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


# ##### Plot of predicted value using the test data compared to the test data

# In[42]:


# Examine the distribution of the predicted values of the testing data.
Title = 'Distribution Plot of Predicted Value Using Testing Data vs Data Distributiono of Test Data'
DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)


# ##### Overfitting
# Overfitting occurs when the model fits the noise, not the underlying process. Therefore when testing your model using the test-set, your model does not perform as well as it is modelling noise, not the underlying process that generated the relationship. Let's create a degree 5 polynomial model.

# In[46]:


# Import packages
from sklearn.preprocessing import PolynomialFeatures

# Create a degree 5 polynomial model, using 55% of the data for testing and the rest for training
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)


# In[47]:


# Perform a degree 5 polynomial transformation on the feature 'horse power'.

# Create a Polynomial with degree 5
pr = PolynomialFeatures(degree=5)

# Fitting/Transforming the x feature in both train & test data
x_train_pr = pr.fit_transform(x_train[["horsepower"]])
x_test_pr = pr.fit_transform(x_test[["horsepower"]])
pr


# In[48]:


# Create a linear regression model "poly" and train it
poly = LinearRegression()

# Fit the Linear Regression model
poly.fit(x_train_pr, y_train)


# In[49]:


# Prediction on Polynomial Regression
yhat = poly.predict(x_test_pr)
yhat[0:5]


# In[50]:


# Taking the first five predicted values and compare it to the actual targets
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)


# In[58]:


# Using the function "PollyPlot" defined at the beginning to display the training data, testing data, and the predicted function
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)


# **Note:** the estimated function appears to track the data but around 200 horsepower, the function begins to diverge from the data points.

# In[59]:


# R^2 of the training data
poly.score(x_train_pr, y_train)


# In[60]:


# R^2 of the test data
poly.score(x_test_pr, y_test)


# **Note: Negative R^2 in the TEST data is a sign of OVERFITTING**

# In[63]:


# How R^2 changes on the test data for different order polynomials and plot the results

Rsque_test = []
order = [1, 2, 3, 4]

# Create a loop throught to generate different degree in the same time
for n in order: 
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[["horsepower"]])
    x_test_pr = pr.fit_transform(x_test[["horsepower"]])
    
    lr.fit(x_train_pr, y_train)
    
    Rsque_test.append(lr.score(x_test_pr, y_test))
    
# Plotting the results
plt.plot(order, Rsque_test)
plt.xlabel("order")
plt.ylabel("R^2")
plt.title("R^2 Using Test Data")
plt.text(3, 0.75, 'Maximum R^2 ')    


# **Note:** R^2 gradually increases until an order three polynomial is used. Then the R^2 dramatically decreases at four

# In[79]:


# Create a function for the interface
def f(order, test_data):
    # Train_Test_Split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    # Define the Polynomial object
    pr = PolynomialFeatures(degree=order)
    
    # Fit & Transform x_train predicted & x_test
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    
    # Define LinearRegression object
    poly = LinearRegression()
    # Fit the model for the new Polynomial Linear Regression 
    poly.fit(x_train_pr,y_train)
    
    # Define the Plot function for the new Polynomial
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)    


# In[80]:


# Create an interface that allows to experiment with different polynomial orders and different amounts of data
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))


# ##### Practice

# In[81]:


# Perform polynomial transformations with more than one feature. Create a "PolynomialFeatures" object "pr1" of degree two
pr1 = PolynomialFeatures(degree=2)


# In[83]:


# Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'
x_train_pr1 = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])


# In[84]:


# x_train_pr1 shape features
x_train_pr1.shape


# In[91]:


# Create a linear regression model "poly1" and train the object using the method "fit" using the polynomial features
poly1= LinearRegression().fit(x_train_pr1, y_train)


# In[93]:


# Use the method "predict" to predict an output on the polynomial features, then use the function "DistributionPlot" to display the distribution of the predicted output vs the test data
yhat_test1 = poly1.predict(x_test_pr1)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)


# ### Ridge regression

# In[94]:


# Perform a degree two polynomial transformation 
pr = PolynomialFeatures(degree=2)

# Fit/Performe & Transform the Polynomial
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])


# In[95]:


# Import package
from sklearn.linear_model import Ridge


# In[96]:


# Create a Ridge regression object, setting the regularization parameter to 0.1
RigeModel = Ridge(alpha=0.1)


# In[97]:


# Fit the model
RigeModel.fit(x_train_pr, y_train)


# In[98]:


# Prediction
yhat = RigeModel.predict(x_test_pr)


# In[100]:


# Compare the first five predicted samples to the test set
print("Predicted:", yhat[0:4])
print("Original test set:", y_test[0:4].values)


# In[ ]:




