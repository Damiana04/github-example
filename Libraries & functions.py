#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[2]:


df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)

print ('Data read into a pandas dataframe!')


# In[3]:


df_can.head()


# In[4]:


df_can.tail()


# In[5]:


df_can.info()


# In[6]:


df_can.columns.values


# In[7]:


# Get a list of indicies
df_can.index.values


# In[8]:


print(type(df_can.columns))
print(type(df_can.index))


# In[9]:


# Get the index and columns as lists 
df_can.columns.tolist()
df_can.index.tolist()

print (type(df_can.columns.tolist()))
print (type(df_can.index.tolist()))


# In[10]:


# Size of dataframe (rows, columns)
df_can.shape 


# In[11]:


# CLEAN the DATA & REMOVE a few UNNECESSARY columns:
 # in pandas axis=0 represents rows (default) and axis=1 represents columns
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
df_can.head(2)


# In[12]:


# RENAME the columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can.columns


# In[13]:


# ADD a 'Total' COLUMN: that sums up the total immigrants by country over the entire period 1980 - 2013
df_can['Total'] = df_can.sum(axis=1)


# In[14]:


# Chech NULL OBJECTS
df_can.isnull().sum()


# In[15]:


# Quick statistical values
df_can.describe


# In[16]:


df_can.describe()


# ####  Select Column in Pandas
# ##### two ways:
# + Method 1: Quick and easy, but only works if the column name does NOT have spaces or special characters. 
# ##### df.column_name 
#         (returns series)
#         
# + Method 2: More robust, and can filter on multiple columns.
# ##### df['column']  
#         (returns series)
#         
#     df[['column 1', 'column 2']] 
#         (returns dataframe)
# 

# In[17]:


# FILTER by METHOD 1: filtering on the list of countries ('Country')
df_can.Country


# In[18]:


# FILTER by METHOD 1: filtering on the list of countries ('OdName') and the data for years: 1980 - 1985
df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]]


# #### Select Row
# ##### 3 ways:
# 1) df.loc[label]        
#         #filters by the labels of the index/column
#         
# 2) df.iloc[index]       
#         #filters by the positions of the index/column       

# In[19]:


df_can.set_index('Country', inplace=True)
# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()


# In[20]:


df_can.head(3)


# In[21]:


# 1. the full row data (all columns)
print(df_can.loc['Japan'])

# alternate methods
print(df_can.iloc[87])
print(df_can[df_can.index == 'Japan'].T.squeeze())


# In[22]:


# 2. for year 2013
print(df_can.loc['Japan', 2013])

# alternate method
print(df_can.iloc[87, 36]) # year 2013 is the last column, with a positional index of 36


# In[23]:


# 3. for years 1980 to 1985
print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])
print(df_can.iloc[87, [3, 4, 5, 6, 7, 8]])


# In[24]:


# Convert the column names into strings: '1980' to '2013'.
df_can.columns = list(map(str, df_can.columns))
# [print (type(x)) for x in df_can.columns.values] #<-- uncomment to check type of column headers


# In[25]:


# PLOTTING 
# useful for plotting later on
years = list(map(str, range(1980, 2014)))
years


# #### Filtering based on a criteria

# In[26]:


# 1. Create the CONDITION boolean SERIES
condition = df_can['Continent'] == 'Asia'
print(condition)


# In[27]:


# 2. Apply this CONDITION into the DATAFRAME
df_can[condition]


# In[28]:


# Apply MULTIPLE CRITERIA in the same line. 
# let's filter for AreaNAme = Asia and RegName = Southern Asia

df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]

# note: When using 'and' and 'or' operators, pandas requires we use '&' and '|' instead of 'and' and 'or'
# don't forget to enclose the two conditions in parentheses


# In[29]:


# OVERVIEW by multiple functions at the same time
print('data dimensions:', df_can.shape)
print(df_can.columns)
df_can.head(2)


# In[30]:


# Using the inline backend

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[31]:


print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0


# #### Line Pots (Series/Dataframe)

# In[32]:


# Extract the data series for Haiti
haiti = df_can.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column
haiti.head()


# In[33]:


haiti.plot()


# In[34]:


haiti.index = haiti.index.map(int) # let's change the index values of Haiti to type integer for plotting
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

plt.show() # need this line to show the updates made to the figure


# In[35]:


# Alternative
years = list(map(str, range(1980, 2010)))
df_can.loc['Haiti', years].plot(kind='bar')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

plt.show()


# In[90]:


# Inserting a tex box in a specific point of the plot, that explanes a specific event
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
plt.text(2000, 6000, '2010 Earthquake') # see note below

plt.show() 


# Quick note on x and y values in plt.text(x, y, label):
# 
# 
#  Since the x-axis (years) is type 'integer', we specified x as a year. The y axis (number of immigrants) is type 'integer', so we can just specify the value y = 6000.
#    
#    plt.text(2000, 6000, '2010 Earthquake') # years stored as type int
#     
#     
# If the years were stored as type 'string', we would need to specify x as the index position of the year. Eg 20th index is year 2000 since it is the 20th year with a base year of 1980.
#     
#     plt.text(20, 6000, '2010 Earthquake') # years stored as type int

# In[96]:


# Compare the number of immigrants from India and China from 1980 to 2013
df_CI = df_can.loc[['India', 'China'], years]
df_CI.head()


# In[97]:


# Calling streight away the plot function, the graph is not right, because indicis & columns should be in the opposite possition
df_CI.plot(kind='line')


# In[98]:


# TURN columns & indices by the function TRANSPOSE
df_CI = df_CI.transpose()
df_CI.head()


# In[99]:


df_CI.plot(kind='line')


# ##### Top 5 countries that contributed the most to immigration to Canada

# In[115]:


# Creating a new dataframe 
df_top_5 = df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)
years = list(map(str, range(1980, 2014)))
df_top_5 = df_can.head()

df_top_5.head()


# In[116]:


# Swiching indicis with columns 
df_top_5 = df_top_5[years].transpose()
df_top_5.head()


# In[139]:


# Visualization by area plot, adjusting size
df_top_5.plot(kind='area', figsize=(14, 8))


# In[136]:


# Visualization by box plot (personal test)
df_top_5.plot(kind='box')


# In[147]:


# Visualization by Bar plot (personal test)
df_top_5.plot(kind='bar', figsize=(14, 8))


# In[148]:


# Visualization by Horizontal Bar plot (personal test)
df_top_5.plot(kind='barh', figsize=(14, 8))


# In[143]:


# Visualization by Histogram plot (personal test)
df_top_5.plot(kind='hist')


# In[149]:


# Visualization by Density plot (personal test)
df_top_5.plot(kind='kde', figsize=(10, 8))


# In[100]:


print(type(haiti))
print(haiti.head(5))


# In[ ]:




