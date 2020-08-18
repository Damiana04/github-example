#!/usr/bin/env python
# coding: utf-8

# #### Exploratory Data Analysis: Sales

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Load data 
Sales = pd.read_csv("Downloads/506_1017_compressed_SalesKaggle3.csv.zip") 
Sales.head()


# ##### Columns description: 
# - File_Type: The value “Active” means that the particular product needs investigation
# - SoldFlag: The value 1 = sale, 0 = no sale in past six months
# - SKU_number: This is the unique identifier for each product
# - Order: Just a sequential counter. Can be ignored
# - SoldFlag: 1 = sold in past 6 mos. 0 = Not sold
# - MarketingType: Two categories of how we market the product
# - New_Release_Flag: Any product that has had a future release (i.e., Release Number > 1)- 

# In[3]:


# Generale info
Sales.info()


# In[4]:


# Statistical values 
Sales.describe()


# In[5]:


# Statistical values, including missing values
Sales.describe(include='all')


# In[6]:


# Shape
Sales.shape


# In[7]:


# Number of unique elelments in each column
Sales.nunique()


# In[8]:


# Exploring File_Type feature
print(Sales["File_Type"])


# In[9]:


# Which unique values are contain into the File_Type feature
Sales["File_Type"].unique() # Discover two categories


# In[10]:


# Unique numbers into File_Type feature
Sales["File_Type"].nunique


# In[11]:


# Count of the 'Historical' and 'Active' state
print(Sales[Sales["File_Type"] == 'Historical']['SKU_number'].count())
print(Sales[Sales["File_Type"] == 'Active']['SKU_number'].count())


# In[12]:


# Dividing/Spliting the Historical & Active data into two separated array
Sales_Historical = Sales[Sales["File_Type"] == "Historical"]
Sales_Active = Sales[Sales["File_Type"] == "Active"]


# In[13]:


# Analysing the Sales_Active array
Sales_Active


# In[14]:


# Which unique values are contain into the File_Type feature
Sales["MarketingType"].unique() # Discover two types of marketing


# In[15]:


# Count values in "MarketingType"
Sales["MarketingType"].value_counts()


# In[16]:


# Exploring percentage between S & D into "MarketingType"
Sales["MarketingType"].value_counts("D", "S")*100


# In[17]:


# Bin for S of MarketingType
bin_S = Sales[Sales["MarketingType"] == 'S'].count()
print(bin_S)


# In[18]:


# Bin for S of MarketingType
bin_D = Sales[Sales["MarketingType"] == 'D'].count
bin_D


# In[19]:


# Plot the values of "MarketingType"
Sales["MarketingType"].value_counts().plot.bar(title="Frequency Distibution of Marketing Type", color='green')


# In[20]:


# Visualization by seaborn displot

# Define column names
col_names = ['StrengthFactor','PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice']

# Define the figure's structure
fig, ax = plt.subplots(len(col_names), figsize=(16,14))
                      
# Loop for filling the plot with data
for i, col_val in enumerate(col_names):
    
    sns.distplot(Sales_Historical[col_val], hist=True, ax=ax[i])
    ax[i].set_title("Frequency Distibution of Marketing Type"+col_val, fontsize=15)
    ax[i].set_xlabel(col_val, fontsize=10)
    ax[i].set_ylabel("Count", fontsize=10)
    
plt.show()


# In[21]:


# Column names
Sales.columns


# In[22]:


# Feature tha we want to exclude
Sales_ins_exclude = Sales[['Order', 'File_Type','SKU_number','SoldFlag','MarketingType','ReleaseNumber','New_Release_Flag']]
Sales_ins_exclude


# In[23]:


# Features that we want to keep
Sales_ins = Sales[['SoldCount', 'StrengthFactor','PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice']]
Sales_ins


# In[24]:


# Checking general info 
Sales_ins.info()


# In[25]:


# Correlation between features into Sales_ins that we want to keep
Sales_ins.corr()


# In[26]:


# Percentage of Correlation between features into Sales_ins that we want to keep
Sales_ins.corr()*100


# In[27]:


# Percentage of Correlation between features into Sales_ins_exclude that we want to delate
Sales_ins_exclude.corr()*100


# In[28]:


# Check if there are any null values in the df Sales
Sales.isnull().values.any()


# In[29]:


# Check null values in each feature
Sales.isnull().sum()


# In[30]:


# Replacing missing values in SoldFlag feature
Sales["SoldFlag"].fillna(0, inplace=True)


# In[31]:


# Replacing missin values in SoldCount feature
Sales["SoldCount"].fillna(0, inplace=True)


# In[32]:


# Check again if there are any others missing values into the df
Sales.isnull().sum()


# ##### Outlier detention analysis

# In[33]:


# Create a loop to detect the outliers for filling the plot

# Define the features where we need to check 
column_names = ['StrengthFactor','PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice']

# Define the plot up/until the end of the columns(len)
fig, ax = plt.subplots(len(column_names), figsize=(8,40))

# FOR (Reterate/Repete) input inserting the values(col_val) into indeces of columns (column_names)
for i, col_val in enumerate(column_names): 
    
    
    # Fill the boxplot as 'y' taking the data from Sales_Historical(old values that we want to explore) & insert them into the indeces [col_val]
    # considering the condition ax[i] means repete the input 'i' up to the end of the columns
    sns.boxplot(y=Sales_Historical[col_val], ax=ax[i])
    ax[i].set_title("Box plot - {}".format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=10)


# **Note**: the black dots are outliers

# ##### Percentile based outlier removal
# The next step that comes to our mind is the ways by which we can remove these outliers. One of the most popularly used technique is the Percentile based outlier removal, where we filter out outliers based on fixed percentile values. The other techniques in this category include removal based on z-score, constant values etc.

# In[34]:


# Define the function for removing the outliers based on percentile
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval) 

col_names = ['StrengthFactor','PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice']

# Define the plot up/until the end of the columns(len)
fig, ax = plt.subplots(len(col_val), figsize=(12, 40))


    # FOR (Reterate/Repete) input inserting the values(col_val) into indeces of columns (column_names)
for i, col_value in enumerate (col_names): 
    # Define x 
    x = Sales_Historical[col_value][:1000]
    sns.distplot(x, ax=ax[i], rug=True, hist=False)
    outliers = x[percentile_based_outlier(x)]
    ax[i].plot(outliers, np.zeros_like(outliers), "ro", clip_on=False) 
        
    ax[i].set_title("Outlier detection - {}".format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=14)
        
plt.show()


# ##### Correlation matrix
# 
# The correlation matrix is a table showing the value of the correlation coefficient (Correlation coefficients are used in statistics to measure how strong a relationship is between two variables. ) between sets of variables. Each attribute of the dataset is compared with the other attributes to find out the correlation coefficient. This analysis allows you to see which pairs have the highest correlation, the pairs which are highly correlated represent the same variance of the dataset thus we can further analyze them to understand which attribute among the pairs are most significant for building the model.

# In[35]:


# Visualizing the Correlation Matrix

# Define the plot measures
f, ax = plt.subplots(figsize=(10,8))

# Define the data for the correlation
corr = Sales_Historical.corr()

# Define the heatmap
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:




