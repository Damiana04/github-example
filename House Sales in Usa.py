#!/usr/bin/env python
# coding: utf-8

# # House Sales in King County, USA

# id : A notation for a house
# 
# date: Date house was sold
# 
# price: Price is prediction target
# 
# bedrooms: Number of bedrooms
# 
# bathrooms: Number of bathrooms
# 
# sqft_living: Square footage of the home
# 
# sqft_lot: Square footage of the lot
# 
# floors :Total floors (levels) in house
# 
# waterfront :House which has a view to a waterfront
# 
# view: Has been viewed
# 
# condition :How good the condition is overall
# 
# grade: overall grade given to the housing unit, based on King County grading system
# 
# sqft_above : Square footage of house apart from basement
# 
# sqft_basement: Square footage of the basement
# 
# yr_built : Built Year
# 
# yr_renovated : Year when house was renovated
# 
# zipcode: Zip code
# 
# lat: Latitude coordinate
# 
# long: Longitude coordinate
# 
# sqft_living15 : Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# 
# sqft_lot15 : LotSize area in 2015(implies-- some renovations)

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the data
file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.describe()


# ###### Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to True

# In[6]:


df.drop('id', axis = 1, inplace = True)
df.drop('Unnamed: 0', axis = 1, inplace = True)
df.describe()


# ##### We can see we have missing values for the columns  bedrooms and  bathrooms 

# In[7]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# ##### We can replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms'  using the method replace(). Don't forget to set the inplace parameter to True

# In[8]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# ##### We also replace the missing values of the column 'bathrooms' with the mean of the column 'bathrooms'  using the method replace(). Don't forget to set the  inplace  parameter top  True 

# In[9]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[11]:


# Verify the new count of NaN
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# ##### Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.

# In[12]:


Unique_floors = df["floors"].value_counts()
Unique_floors.to_frame()


# ##### Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# In[13]:


sns.boxplot(x= "waterfront", y= "price", data = df)


# ##### Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.

# In[14]:


sns.regplot(x= "sqft_above", y= "price", data = df)


# ###### We can use the Pandas method corr() to find the feature other than price that is most correlated with price.

# In[15]:


df.corr()['price'].sort_values()


# ## Model Development
# 
# #### We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.

# In[25]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# ##### Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2.

# In[26]:


X1 = df[['sqft_living']]
Y1 = df['price']
lm = LinearRegression()
lm
lm.fit(X1,Y1)
lm.score(X1, Y1)


# ##### Fit a linear regression model to predict the 'price' using the list of features:

# In[19]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# ###### Then calculate the R^2. Take a screenshot of your code.

# In[20]:


X2 = df[features]
Y2 = df['price']
lm.fit(X2,Y2)
lm.score(X2,Y2)


# Create a list of tuples, the first element in the tuple contains the name of the estimator:
# 
# 'scale'
# 
# 'polynomial'
# 
# 'model'
# 
# The second element in the tuple contains the model constructor
# 
# StandardScaler()
# 
# PolynomialFeatures(include_bias=False)
# 
# LinearRegression()

# In[21]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# ##### Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2.

# In[28]:


pipe=Pipeline(Input)
pipe


# In[29]:


pipe.fit(X,Y)


# In[30]:


pipe.score(X,Y)


# ## Model Evaluation and Refinement

# In[32]:


# Import the necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[33]:


# Split the data into training and testing sets
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# ##### Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.

# In[34]:


# Import 
from sklearn.linear_model import Ridge


# In[35]:


RigeModel = Ridge(alpha=0.1) 
RigeModel.fit(x_train, y_train)
RigeModel.score(x_test, y_test)


# ##### Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.

# In[36]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

RigeModel = Ridge(alpha=0.1) 
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)


# In[ ]:




