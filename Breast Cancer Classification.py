#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer


# In[3]:


# Dictionaries & Columns
cancer.keys()


# In[4]:


# Description feature
print(cancer['DESCR'])


# In[5]:


# Target feature
print(cancer['target'])


# In[6]:


# Target Names feature
print(cancer['target_names'])


# In[7]:


# Filename feature
print(cancer['filename'])


# In[8]:


# Feature Names feature
print(cancer['feature_names'])


# In[9]:


# Data feature
print(cancer['feature_names'])


# In[10]:


# Data shape
cancer['data'].shape


# In[11]:


# Creating the dataframe 
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()


# In[12]:


# The bottom of the dataframe
df_cancer.tail()


# ### Visualizing the data

# In[13]:


# Pair plot
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])


# In[14]:


# Pair plot
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])


# In[15]:


# Count plot
sns.countplot(df_cancer['target'])


# In[16]:


# Scatter plot
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[17]:


# Correlation
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot = True)


# ## Training the Model

# In[18]:


# Defining the features/X
X = df_cancer.drop(['target'], axis = 1)
X


# In[19]:


# Define target/y
y = df_cancer['target']
y


# In[20]:


# Train, Test & Split

# Import package
from sklearn.model_selection import train_test_split

# Define the function
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2 , random_state=5)


# In[21]:


# Test&Train shapes for X,y
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[22]:


# Import SVM package for Training the model
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# In[23]:


# Define Support Vector Machine object
svc_model = SVC()


# In[24]:


# Fit/Train the model
svc_model.fit(X_train, y_train)


# ## Evaluating the Model

# In[25]:


# Prediction
y_predict = svc_model.predict(X_test)


# In[26]:


y_predict


# In[27]:


# Confusion matrix
cm = confusion_matrix(y_test, y_predict)


# In[28]:


# Confusion matrix visualization
sns.heatmap(cm, annot = True)


# ## Improving the Model

# In[29]:


# Normalizing the Train dataset: Finding the Minimum Value
min_train = X_train.min()


# In[30]:


# Normalizing the Train dataset: Creating the range
range_train = (X_train-min_train).max()


# In[31]:


# Normalizing the Train dataset: Scaling 
X_train_scaled = (X_train-min_train)/range_train


# In[32]:


# Visualizing the Train dataset before Normalization
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[33]:


# Visualizing the Train dataset after Normalization with the right range
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[34]:


# Normalizing the Test dataset: Finding the Minimum Value
min_test = X_test.min()


# In[35]:


# Normalizing the Test dataset: Creating the range
range_test = (X_test-min_test).max()


# In[36]:


# Normalizing the Test dataset: Scaling
X_test_scaled = (X_test-min_test)/range_test


# In[37]:


# Fit/Train the model
svc_model.fit(X_train_scaled, y_train)


# In[38]:


# Visualizing the Test dataset before Normalization
sns.scatterplot(x = X_test['mean area'], y = X_test['mean smoothness'], hue = y_test)


# In[39]:


# Visualizing the Train dataset after Normalization with the right range
sns.scatterplot(x = X_test_scaled['mean area'], y = X_test_scaled['mean smoothness'], hue = y_test)


# In[40]:


# Training the Model with the Scaled dataset
svc_model.fit(X_train_scaled, y_train)


# In[41]:


# Prediction 
y_predict = svc_model.predict(X_test_scaled)


# In[42]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)


# In[43]:


# Visualization of the Confusion Matrix
sns.heatmap(cm, annot=True)


# In[44]:


# Classification Reports
print(classification_report(y_test, y_predict))


# In[45]:


# Improving the Model by Grid: C & Gamma parameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}


# In[46]:


# Importing Grid package
from sklearn.model_selection import GridSearchCV


# In[47]:


# Create the Grid statement
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)


# In[48]:


# Fit/Train the model
grid.fit(X_train_scaled, y_train)


# In[49]:


# Best parameters' Grid
grid.best_params_


# In[50]:


# Predicting Test dataset
grid_predictions = grid.predict(X_test_scaled)


# In[51]:


# Confusion Matrix
cm = confusion_matrix(y_test, grid_predictions)


# In[52]:


# Visualization of the Confusion Matrix
sns.heatmap(cm, annot = True)


# In[53]:


# Classification Reports
print(classification_report(y_test, grid_predictions))


# In[ ]:





# In[ ]:




