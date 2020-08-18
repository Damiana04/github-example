#!/usr/bin/env python
# coding: utf-8

# #### House prices forecasting model

# In[1]:


# Import packages
import numpy as np
import pandas as pd

# Installing plotly
get_ipython().system('pip install plotly==4.9.0')


# Data viz libs
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go

# Sklearn libs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Models Libs
get_ipython().system('pip install xgboost')
from xgboost import XGBRegressor

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# Load TRAIN dataset
df_train = pd.read_csv("Downloads/train_house(1).csv")
df_train.head()


# In[3]:


# Train dataset Shape
df_train.shape


# In[4]:


# Load TEST dataset
df_test = pd.read_csv("Downloads/test_house(2).csv")
df_test.head()


# In[5]:


# Test dataset Shape
df_test.shape


# ##### Hypotheses
# - [X] The **seasons don't impact** the price.
# - [X] The **lot area don't impact** the price.

# ##### EDA - Exploratory Data Analysis
# I create this new data frame, where I selected the columns 'SalePrice' and 'MoSold'.
# The idea is 'Use the month's to get the seasons and using boxplot, compare the distribuiton of the house's prices by seasons.

# In[6]:


# Create the new df
year_seasons_df = df_train[['SalePrice','MoSold']].copy()
year_seasons_df.head()


# In[7]:


# Divide the months trought the function
def setSeason(month):
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    if month in (12, 1, 2):
        return "Winter"
    return "Spring"

year_seasons_df['yearSeason'] = year_seasons_df.MoSold.apply(lambda x: setSeason(x));

year_seasons_df.sort_values(by='SalePrice', inplace=True)

trace = go.Box(
    x = year_seasons_df.yearSeason,
    y = year_seasons_df.SalePrice
)

data = [trace]

layout = go.Layout(title="Prices x Year Season",
                  yaxis={'title':'Sale Price'},
                  xaxis={'title':'Year Season'})

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[8]:


year_seasons_gp_df = year_seasons_df.groupby("yearSeason")["SalePrice"].count().reset_index()

year_seasons_gp_df = pd.DataFrame({'yearSeason': year_seasons_gp_df.yearSeason, 'CountHouse': year_seasons_gp_df.SalePrice}) 

year_seasons_gp_df.sort_values(by='CountHouse', inplace=True)


# In[9]:


# Count how many houses were sold by year station
trace= go.Bar(x=year_seasons_gp_df.yearSeason, y=year_seasons_gp_df.CountHouse)

data = [trace]

layout = go.Layout(title="Count House x Year Season",
                  yaxis={'title':'Count House'},
                  xaxis={'title':'Year Season'})

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)


# In[10]:


# Correlation setting

def labelSeason(x): 
    if x == "Summer": 
        return 1
    if x == "Autumn":
        return 2
    if x == "Winter":
        return 3
    return 4

year_seasons_df["labelSeason"] = year_seasons_df.yearSeason.apply(lambda x: labelSeason(x))

df_corr_year_seasons = year_seasons_df.corr()
df_corr_year_seasons


# In[11]:


year_seasons_sorted_df= year_seasons_df.sort_values(by='MoSold')

year_seasons_sorted_gp_df = year_seasons_df.groupby('MoSold')["SalePrice"].count().reset_index()


# In[12]:


year_seasons_sorted_df


# In[13]:


year_seasons_sorted_df.count()


# In[14]:


seasonality = year_seasons_sorted_df.groupby('yearSeason').count()
seasonality


# In[15]:


seasonality.sort_values(by='yearSeason').count()


# In[16]:


seasonality.groupby('yearSeason').count()


# In[17]:


seasonality.groupby('yearSeason')['SalePrice'].count()


# In[18]:


seasonality.sort_values(by='yearSeason')[['SalePrice']]


# In[19]:


seasonality.sort_values(by='SalePrice').count()


# In[20]:


year_seasons_sorted_gp_df


# In[21]:


df_Utilities = df_train[["Utilities"]]
df_Utilities.info()


# In[22]:


df_Utilities.tail


# In[23]:


df_train['Utilities'].unique()


# In[24]:


df_train.head()


# In[25]:


SaleCondition = df_train[['SaleCondition']]
SaleCondition


# In[26]:


SaleCondition['SaleCondition'].unique()


# In[27]:


SaleCondition.nunique


# In[28]:


print("Normal:", SaleCondition[SaleCondition['SaleCondition'] == 'Normal'].count())
print('Abnorml:',SaleCondition[SaleCondition['SaleCondition'] == 'Abnorml'].count())
print("Partial:",SaleCondition[SaleCondition['SaleCondition'] == 'Partial'].count())
print("AdjLand:",SaleCondition[SaleCondition['SaleCondition'] == 'AdjLand'].count())
print("Alloca:",SaleCondition[SaleCondition['SaleCondition'] == 'Alloca'].count())
print("Family:",SaleCondition[SaleCondition['SaleCondition'] == 'Family'].count())


# In[29]:


SaleCondition_2 = SaleCondition.copy()
SaleCondition_2


# ##### Sales by month

# In[30]:


# Visualizing how many houeses were sold by month
year_seasons_sorted_df = year_seasons_df.sort_values(by='MoSold')

year_seasons_sorted_gp_df = year_seasons_df.groupby('MoSold')['SalePrice'].count().reset_index()

df = year_seasons_sorted_gp_df

trace = go.Scatter(x=df.MoSold, y=df.SalePrice, mode='markers+lines', line_shape='spline')

data = [trace]

layout = go.Layout(title='Sales by month', yaxis={'title':'Count House'}, xaxis={'title':'Month sold', 'zeroline':False})

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ##### Lot Area by Price

# In[31]:


# Visualization
trace = go.Scatter(x=df_train.LotArea, y=df_train.SalePrice, mode= 'markers')

data = [trace]

layout = go.Layout(title='Lot Area x Sale Price', yaxis= {'title':'Sale Price'}, xaxis={'title':'Lot Area'})

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)


# ##### Distribution of Sale Price

# In[32]:


# Visualization of Distribution of Sale Price
trace = go.Box(y=df_train.SalePrice, name='Sale Price')

data= [trace]

layout = go.Layout(title="Distribution Sale Price", yaxis={'title':'Distribution Sale Price'})

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ##### Distribution of lot area

# In[33]:


# Visualization of Distribution of lot area
trace = go.Box(y=df_train.LotArea, name= 'Lot Area')

data = [trace]

layout = go.Layout(title="Distribuiton Lot Area", yaxis={'title':'Lot Area'})

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ##### Correlation between Lot Area and Price

# In[34]:


# Correlation between Lot Area & Price
df_LotArea_Price = df_train[["LotArea", "SalePrice"]]
df_LotArea_Price.corr()


# ##### Removing Outiliers

# In[35]:


# Removing the outliers greater then 700K SalePrice
df_train = df_train.drop(df_train.loc[(df_train['LotArea'] > 70000)].index)
df_train = df_train.drop(df_train.loc[(df_train['SalePrice'] > 500000)].index)


# ##### New distribution Sale Price & Lot Area

# In[36]:


# Visualization of Sale Price & Lot Area after removing outliers
trace = go.Scatter(
    x = df_train.LotArea,
    y = df_train.SalePrice,
    mode = 'markers'
)

data = [trace]

layout = go.Layout(title="Lot Area x Sale Price",
                  yaxis={'title':'Sale Price'},
                  xaxis={'title':'Lot Area'})

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)


# ###### Targets and features

# In[37]:


# Define y (target/dependent variable) as SalePrice
y = np.log(df_train.SalePrice)

# Define X (features/indipendent variables) as all of others features in the TRAIN dataset 
X = df_train.copy()

# Define X (features/indipendent variables) as all of others features in the TEST dataset
X_test = df_test.copy()


# ##### Feature engineering

# In[38]:


# Feature engineering in the Train dataset
X["AreaUtil"] = X["LotArea"] - X["MasVnrArea"] + X["GarageArea"] + X["PoolArea"]

# Feature engineering in the Test dataset
X_test["AreaUtil"] = X_test["MasVnrArea"] - X_test["GarageArea"] + X_test["PoolArea"]


# In[39]:


# The Have Pool it's a boolean feature, if the pool area it's greater than 0 means that house have a pool

# Create a new dataframe in the Train dataset for "HavePool"
X["HavePool"] = X["PoolArea"] > 0

# Create a new dataframe in the Test dataset for "HavePool"
X_test["HavePool"] = X_test["PoolArea"] > 0


# In[43]:


# Squared for Train dataset
X["GarageCars2"] = X["GarageCars"]**2
X["GarageCarsSQRT"] = np.sqrt(X["GarageCars"])
X["GarageArea2"] = X["GarageArea"]**2
X["GarageAreaSQRT"] = np.sqrt(X["GarageCars"])
X["LotArea2"] = X["LotArea"]**2
X["LotArea2SQRT"] = np.sqrt(X["LotArea"])
X["AreaUtil2"] = X["AreaUtil"]**2
X["AreaUtilSQRT"] = np.sqrt(X["AreaUtil"])
X["GrLivArea2"] = X["GrLivArea"]**2
X["GrLivAreaSQRT"] = np.sqrt(X["GrLivArea"])

# Squared for Test dataset
X_test["GarageCars2"] = X_test["GarageCars"]**2
X_test["GarageCarsSQRT"] = np.sqrt(X_test["GarageCars"])
X_test["GarageArea2"] = X_test["GarageArea"]**2
X_test["GarageAreaSQRT"] = np.sqrt(X_test["GarageCars"])
X_test["LotArea2"] = X_test["LotArea"]**2
X_test["LotArea2SQRT"] = np.sqrt(X_test["LotArea"])
X_test["AreaUtil2"] = X_test["AreaUtil"]**2
X_test["AreaUtilSQRT"] = np.sqrt(X_test["AreaUtil"])
X_test["GrLivArea2"] = X_test["GrLivArea"]**2
X_test["GrLivAreaSQRT"] = np.sqrt(X_test["GrLivArea"])


# In[44]:


# Correlation's visualization in Sale Price
corrmat = X.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"]) > 0]

# Most correlated features
if 1 == 1:
    plt.figure(figsize=(30, 15))
    g = sns.heatmap(X[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# In[45]:


# Remove row with missing target
X.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Drop target
X.drop(["SalePrice"], axis=1, inplace=True)

X.drop(["OverallQual"], axis=1, inplace=True)


# In[46]:


X.head()


# In[ ]:


# Visualization 
x.he

