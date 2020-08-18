#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis

# In[2]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system(' pip install seaborn')


# In[3]:


path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# In[4]:


# list the data types for each column
print(df.dtypes)


# In[5]:


df.info


# In[6]:


# Missing values
missing_data = df.isnull()
missing_data.head()


# In[7]:


# Finding missing values
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[9]:


# Chech NULL OBJECTS
df.isnull().sum()


# ### CORRELATION between ariables of type "int64" or "float64" using the method "corr":

# In[9]:


df[['bore','stroke' ,'compression-ratio','horsepower']].corr()


# ##   Positive linear relationship

# In[10]:


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
df[["engine-size", "price"]].corr()


# ##   Negative linear relationship

# In[11]:


# Highway-mpg as potential predictor variable of price
sns.regplot(x="highway-mpg", y="price", data=df)
df[["highway-mpg", "price"]].corr()


# ##   Weak linear relationship

# In[12]:


# Peak-rpm as potential predictor variable of price
sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm','price']].corr()


# In[13]:


# Stroke as potential predictor variable of price
sns.regplot(x="stroke", y="price", data=df)
df[["stroke","price"]].corr()


# ### Categorical variables

# In[14]:


# Relationship between "body-style" and "price"
sns.boxplot(x="body-style", y="price", data=df)


# In[15]:


# Relationship between "engine-location" and "price"
sns.boxplot(x="engine-location", y="price", data=df)


# In[16]:


# Relationship between "engine-location" and "price"
sns.boxplot(x="drive-wheels", y="price", data=df)


# ## Descriptive Statistical Analysis

# In[17]:


df.describe()


# In[18]:


df.describe(include=['object'])


# In[19]:


#Counting value into a specific feature/column
df['drive-wheels'].value_counts()


# In[20]:


# Convert the series to a Dataframe
df['drive-wheels'].value_counts().to_frame()


# In[21]:


# Repeating the step above. Saving results to the dataframe & rename the column 
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# In[22]:


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head()


# ###   Basics of Grouping

# In[23]:


# Finding unique categories into a specific feature/category
df['drive-wheels'].unique()


# In[24]:


df['drive-wheels'].unique


# In[25]:


#If we want to know, on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them.

df_group_one = df[['drive-wheels','body-style','price']]

# Grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one

# Grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


# ###       Create a Pivot table

# In[26]:


grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[27]:


# REPLACING MISSING VALUES IN A PIVOT TABLE
  # Filling missing values with 0
    
grouped_pivot = grouped_pivot.fillna(0) 
grouped_pivot


# In[28]:


# grouping results
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle


# ### HEAT MAP: Variables: Drive Wheels and Body Style vs Price

# In[29]:


# Use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[30]:


# LABELING EACH AXIS IN THE HEAT MAP:

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# #  Correlation and Causation
# ####     Correlation: a measure of the extent of interdependence between variables.
# ####     Causation: the relationship between cause and effect between two variables.

# ##  Pearson Correlation
# ###### measures the linear dependence between two variables X and Y
# ###### coefficient is a value between -1 and 1 inclusive
# + 1: Total positive linear correlation.
# + 0: No linear correlation, the two variables most likely do not affect each other.
# + -1: Total negative linear correlation.
# 
# ###### Pearson Correlation is the default method of the function "corr"

# In[31]:


df.corr()


# ### P-value: is the probability value that the correlation between these two variables is statistically significant. 
# ###### significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.
# + p-value is  <  0.001: we say there is strong evidence that the correlation is significant.
# + p-value is  <  0.05: there is moderate evidence that the correlation is significant.
# + p-value is  <  0.1: there is weak evidence that the correlation is significant.
# + p-value is  >  0.1: there is no evidence that the correlation is significant.

# In[32]:


from scipy import stats


# In[42]:


# Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'
pearson_coef,p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# ##### Conclusion:
# Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585)

# In[34]:


# Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'
pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# ######   Conclusion:
# Since the p-value is  <  0.001, the correlation between horsepower and price is statistically significant, and the linear relationship is quite strong (~0.809, close to 1)

# In[35]:


# Pearson Correlation Coefficient and P-value of 'length' and 'price'
pearson_coef, p_value = stats.pearsonr(df["length"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# ######  Conclusion:
# Since the p-value is  <  0.001, the correlation between length and price is statistically significant, and the linear relationship is moderately strong (~0.691).

# In[36]:


# Pearson Correlation Coefficient and P-value of 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df["width"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# ######  Conclusion:
# Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751).

# In[37]:


# Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df["curb-weight"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# ######  Conclusion:
# Since the p-value is  <  0.001, the correlation between curb-weight and price is statistically significant, and the linear relationship is quite strong (~0.834).

# In[38]:


# Pearson Correlation Coefficient and P-value of 'engine-size' and 'price'
pearson_coef, p_value = stats.pearsonr(df["engine-size"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# ######  Conclusion:
# Since the p-value is  <  0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).

# In[39]:


# Pearson Correlation Coefficient and P-value of 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df["bore"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# #####  Conclusion:
# Since the p-value is  <  0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).

# In[40]:


# Pearson Correlation Coefficient and P-value of 'city-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df["city-mpg"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# #####  Conclusion:
# Since the p-value is  <  0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of ~ -0.687 shows that the relationship is negative and moderately strong.

# In[41]:


# Pearson Correlation Coefficient and P-value of 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df["highway-mpg"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P=", p_value)


# #####  Conclusion:
# Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of ~ -0.705 shows that the relationship is negative and moderately strong.

# #    ANOVA: Analysis of Variance
# statistical method used to test whether there are significant differences between the means of two or more groups. 
# 
# ANOVA returns two parameters:
# 
# ##### F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.
# 
# ##### P-value: P-value tells how statistically significant is our calculated score value.
# 
# If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value.

# In[44]:


# Different types 'drive-wheels' impact 'price'. 
# Group the data
grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)


# In[45]:


df_gptest


# In[46]:


grouped_test2.get_group('4wd')['price']


# In[47]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[49]:


# Separately: fwd and rwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val )


# In[50]:


# 4wd and rwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])    
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[51]:


# 4wd and fwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])   
print("ANOVA results: F=", f_val, ", P =", p_val)


# In[ ]:




