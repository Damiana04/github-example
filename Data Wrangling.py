#!/usr/bin/env python
# coding: utf-8

# ### Data Wrangling

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# In[2]:


# File
filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"


# In[3]:


# Headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[4]:


# Load data
df = pd.read_csv(filename, names = headers)
df.head()


# #### Steps for working with MISSING data
# - dentify missing data
# - deal with missing data
# - correct data format

# #### Identify missing values
# 
# 
# #### Convert "?" to NaN

# In[5]:


# Replace "?" to NaN
df.replace("?", np.nan, inplace=True)
df.head()


# ##### Evaluating for Missing Data

# In[6]:


# Counting missing data by ISNULL (opposite boolean result respect to NOTNULL)
missing_data = df.isnull()
missing_data.head()


# In[7]:


# Counting missing data by NOTNULL (opposite boolean result respect to ISNULL)
missing_data = df.notnull()
missing_data.head()


# ##### Count missing values in each column

# In[8]:


# Loop through show missing data in each columns
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# <h3 id="deal_missing_values">Deal with missing data</h3>
# 
# <ol>
#     <li>Drop data<br>
#         a. drop the whole row<br>
#         b. drop the whole column
#     </li>
#     <li>Replace data<br>
#         a. replace it by mean<br>
#         b. replace it by frequency<br>
#         c. replace it based on other functions
#     </li>
# </ol>

# #### Replace by AVG

# In[9]:


# Calculate the average of the column "normalized-losses" & converting values to float
avg_norm_loss = df["normalized-losses"].astype('float').mean(axis=0)
print("The avg for normalized-losses is:", avg_norm_loss)


# In[10]:


# Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df["normalized-losses"].head


# In[11]:


# Calculate the mean value for 'bore' column & converting values to float
avg_bore = df['bore'].astype('float').mean(axis=0)
print("The avg for bore is:", avg_bore)


# In[12]:


# Replace "NaN" by mean value in "bore" column
df["bore"].replace(np.nan, avg_bore, inplace=True)
df["bore"].head


# In[13]:


# Calculate the mean value for 'stroke' column & converting values to float
avg_stroke = df["stroke"].astype('float').mean(axis=0)
print("The avg stroke is:", avg_stroke)


# In[14]:


# Replace "NaN" by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
df["stroke"].head


# In[15]:


# Calculate the mean value for 'horsepower' column & converting values to float
avg_horsepower = df["horsepower"].astype('float').mean(axis=0)
print("The avg horsepoweris:", avg_horsepower)


# In[16]:


# Replace "NaN" by mean value in "horsepower" column
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)
df["horsepower"].head


# In[17]:


# Calculate the mean value for 'peak-rpm' column & converting values to float
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("The avg peak rpm is:", avg_peakrpm)


# In[18]:


# Replace "NaN" by mean value in "peak-rpm" column
df["peak-rpm"].replace(np.nan, avg_peakrpm, inplace=True)
df["peak-rpm"].head


# In[19]:


# Calculate the mean value for 'price' column & converting values to float
avg_price = df["price"].astype('float').mean(axis=0)
print("The avg price is:", avg_price)


# In[20]:


# See which values are present in a particular column
df["num-of-doors"].value_counts()


# In[21]:


# Calculate the most common type automatically
df["num-of-doors"].value_counts().idxmax()


# ##### Replace by most common value

# In[22]:


# Replace missing data in 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)
df["num-of-doors"].head


# In[23]:


missing_data = df.isnull()
missing_data.head(5)


# In[24]:


# Check which other features contain missin values

# Loop through show missing data in each columns
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[25]:


# Drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# Reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# ##### Correct data format
# 
# In Pandas:
# <p><b>.dtype()</b> to check the data type</p>
# <p><b>.astype()</b> to change the data type</p>

# In[26]:


# Check the data type
df.dtypes


# In[27]:


# Converting data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype('float')
df[["normalized-losses"]] = df[["normalized-losses"]].astype('int')
df[["price"]] = df[["price"]].astype('float')
df[["peak-rpm"]] = df[["peak-rpm"]].astype('float')


# In[28]:


# Check the results
df.dtypes


# ##### Data Standardization

# In[29]:


# Transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# Check transformed data 
df.head()


# ##### Data Normalization
# 
# Normalization is the process of transforming values of several variables into a similar range

# In[30]:


# Replace (original value) by (original value)/(maximum value) = AVG
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df.head()


# ##### Binning

# In[31]:


# Convert data to correct format
df[["horsepower"]] = df[["horsepower"]].astype(int, copy=True)


# In[32]:


# Plotting values

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# Plotting by histogram
plt.pyplot.hist(df["horsepower"])
# Set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("Binning horsepower")


# In[33]:


# Building 3 bins of equal length, there should be 4 dividers, so numbers_generated=4 
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[34]:


# Set group names
group_names = ['Low', 'Medium', 'High']


# In[35]:


# Determine what each value of "df['horsepower']" belongs to (.cut function)
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)
df[['horsepower','horsepower-binned']].head(20)


# In[36]:


# Number of vehicles in each bin
df["horsepower-binned"].value_counts()


# In[37]:


# Plotting the distribution of each bin

# Import packages
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# Plotting by barchart
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# Set x/y labels and plot title
pyplot.xlabel("horsepower")
pyplot.ylabel("count")
pyplot.title("Horsepower Bins")


# ###### Bins visualization

# In[38]:


# Import packages
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# Define plots
a = (0,1,2)

# Draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins=3)

# Set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot. title("Horsepower Bins")


# ##### Indicator variable (or dummy variable)

# In[39]:


# Check columns
df.columns


# In[40]:


# Get indicator variables and assign it to data frame "dummy_variable_1
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[41]:


# Change column names for clarity
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()


# In[42]:


# Merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)


# In[43]:


# Drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# ##### Practice: 
# Create indicator variable to the column of "aspiration": "std" to 0, while "turbo" to 1.

# In[44]:


# Get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# Change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# Show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# In[45]:


# Merge the new dataframe to the original dataframe then drop the column 'aspiration'
df = pd.concat([df, dummy_variable_2], axis=1)


# In[46]:


# Drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


# In[47]:


# Save the new csv
df.to_csv('clean_df.csv')


# In[ ]:




