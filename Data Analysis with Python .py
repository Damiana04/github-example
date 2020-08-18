#!/usr/bin/env python
# coding: utf-8

# ### Data Analysis with Python

# In[2]:


# Import pandas library
import pandas as pd

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)
df.head()


# In[3]:


# Checking the bottom 10 rows of data frame "df"
df.tail()


# #### Add Headers 

# In[7]:


# Create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)


# In[11]:


# Check the result after havind added columns names
df.columns = headers
df.head()


# In[12]:


# Drop missing values into the rows(axis=0) in the column "price"
df.dropna(subset=["price"], axis=0)


# In[14]:


# Get the column names
df.columns


# #### Save Dataset

# In[15]:


# Saving my new dataframe in a csv
df.to_csv("automobile.csv", index=False)


# #### Basic Insight of Dataset

# In[16]:


# Data types
df.dtypes


# #### Statistical values

# In[20]:


# Describe 5 rows on top & 5 rows on bottom
df.describe


# In[23]:


# Describe the first 5 rows & the first columns
df.describe()


# In[22]:


# Describe All the columns in "df" 
df.describe(include = "all")


# #### Selecting columns & analyising values

# In[24]:


# Selecting specific columns
df[['length', 'compression-ratio']]


# In[27]:


# Describing 5 rows on top & 5 rows on bottom of two specific columns
df[['length', 'compression-ratio']].describe


# In[28]:


# Describing the first 5 rows of two specific columns
df[['length', 'compression-ratio']].describe()


# #### Info

# In[31]:


# Check values in the df 5 rows on top & 5 rows on bottom
df.info


# In[33]:


# Check the values for each columns
df.info()


# In[ ]:




