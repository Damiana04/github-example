#!/usr/bin/env python
# coding: utf-8

# ### Predictive analysis of Bank Marketing
# 
# What to achieve?
# The classification goal is to predict if the client will subscribe a term deposit (variable y).

# In[ ]:


https://www.kaggle.com/kevalm/predictive-bank-marketing-analysis


# In[1]:


# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

#Classification Algorithms 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as m
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load dataset
data = pd.read_csv("bank.csv", delimiter=";",header='infer')
data.head()


# In[3]:


# Shape
data.shape


# In[4]:


# Statistical values, including NaN values
data.describe(include='all')


# In[5]:


# Checking missing values in each column
data.isnull().sum()


# In[6]:


# General info
data.info()


# ##### Finding correlation between features and class for selection

# In[7]:


# Visualization of the correlation between features
sns.pairplot(data)


# Note that data here is not-symmetric. So lets find out the correlation matrix to look into details

# In[8]:


# Correlation matrix
data.corr()


# In[9]:


# Visualization of correlation 
corr = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True)


# ##### Observations from pairplot, correlation matrix, and heatmap:
# 
# - Data is non-linear. Asymmetric.
# - Hence selection of features will not depend upon correlation factor.
# - Also not a single feature is correlated completely with class, hence requires combinantion of features.

# ##### Feature Selection techniques:
# 
# - Univariate Selection (non-negative features). 
# - Recursive Feature Elimination (RFE). 
# - Principal Component Analysis (PCA) (data reduction technique). 
# - Feature Importance (decision trees). 

# ##### Which feature selection technique should be used for the data?
# 
# - Contains negative values, hence Univariate Selection technique cannot be used.
# - PCA is data reduction technique. Aim is to select best possible feature and not reduction and this is classification type of data.
# - PCA is an unsupervised method, used for dimensionality reduction.
# - Hence Decision tree technique and RFE can be used for feature selection.
# - Best possible technique will be which gives extracts columns who provide better accuracy.

# ##### Encoding Categorical and numerical data into digits form

# In[10]:


# Checking datatypes
data.dtypes


# ##### Converting object type data into One-Hot Encoded data using get_dummies method

# In[11]:


# Get_dummies
data_new = pd.get_dummies(data, columns=['job','marital','education','default', 'housing','loan', 'contact','month', 'poutcome'])


# In[12]:


# Check the result
data_new.dtypes


# In[13]:


# Checking the original dataset
data.dtypes


# In[14]:


# Classifying column into binary format
data_new.y.replace(("yes","no"), (1,0), inplace=True)


# In[15]:


# Checking the result
data_new.dtypes


# ##### Exploring features

# In[16]:


# Columns 
data.columns


# In[17]:


# Education feature
data.education.unique()


# In[18]:


# y feature
data.y.unique()


# In[19]:


# Using Crosstab to display education stats respect to y ie class variable
pd.crosstab(index=data["education"], columns=data["y"])


# In[20]:


# Education categories and there frequency
data.education.value_counts().plot(kind='barh')


# ##### Classifiers : Based on the values of different parameters we can conclude to the following classifiers for Binary Classification.
# 1. Gradient Boosting
# 2. AdaBoosting
# 3. Logistics Regression
# 4. Random Forest Classifier
# 5. Linear Discriminant Analysis
# 6. K Nearest Neighbour
# 7. Decision Tree
# 8. Gaussian Naive Bayes 
# 9. Support Vector Classifier

# In[21]:


# Import packages
from xgboost import XGBClassifier


# In[22]:


# Define Classifiers
classifiers = {'Adaptive Boosting Classifier':AdaBoostClassifier(),
               'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),
               'Logistic Regression':LogisticRegression(),
               'Random Forest Classifier': RandomForestClassifier(),
               'K Nearest Neighbour':KNeighborsClassifier(8),
               'Decision Tree Classifier':DecisionTreeClassifier(),
               'Gaussian Naive Bayes Classifier':GaussianNB(),
               'Support Vector Classifier':SVC(),
              }


# In[23]:


# Due to one hot encoding increase in the number of columns
data_new.shape


# In[24]:


# Create the new dataframes for X & y 
data_y = pd.DataFrame(data_new['y'])
data_X = data_new.drop(['y'], axis=1)

# Check the results
print(data_y.columns)
print(data_X.columns)


# In[25]:


# Define the metrics
log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)


# In[26]:


# Import package to ignore errors
import warnings
warnings.filterwarnings('ignore')


# In[34]:


# Define Stratified Shuffle Split statement
rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=2)
rs.get_n_splits(data_X, data_y)

# Define the loop for Train_Test_Split data
for Name,classify in classifiers.items():
    for train_index, test_index in rs.split(data_X, data_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X,X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        clf = classify
        cls = clf.fit(X,y)
        y_out = clf.predict(X_test)
        accuracy = m.accuracy_score(y_test, y_out)
        precision = m.precision_score(y_test, y_out, average='macro')
        recall = m.recall_score(y_test, y_out, average = 'macro')
        # roc_auc = roc_auc_score(y_out,y_test)
        f1 = m.f1_score(y_test, y_out, average = 'macro')
        log_entry = pd.DataFrame([[Name, accuracy, precision, recall, roc_auc, f1]], columns=log_cols)
        # metric_entry = pd.DataFrame([[precision, recall, roc_auc, f1]], columns=metrics_cols)
        log = log.append(log_entry)
        # metric = metric.append(metric_entry)
        
        
print(log)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes('muted')
sns.barplot(x='Accuracy', y='Classifier', data=log, color='g')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




