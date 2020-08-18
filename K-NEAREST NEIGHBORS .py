#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading data
df = pd.read_csv('Downloads/teleCust1000t.csv')
df.head()


# ##### Data Visualization and Analysis

# In[4]:


# Counting values in the dataset
df['custcat'].value_counts()


# In[5]:


# Visualizzation of "income" by histogram
df.hist(column='income', bins=50)


# ##### Feature set: defining X & y

# In[6]:


df.columns


# In[7]:


# Use scikit-learn library, we have to convert the Pandas data frame to a Numpy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# In[8]:


# Defining y & converting labels
y = df['custcat'].values
y[0:5]


# ##### Normalize Data
# 
# Data Standardization give data zero mean and unit variance.
# It is good practice, especially for algorithms such as KNN which is based on distance of cases

# In[9]:


# Fit & Transform data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# ##### OUT-of-SAMPLE-ACCURACY
# is the percentage of the correct prediction that the model has NOT been trained on. 
# 
# IMPROVE the Out-of-Sample-Accuracy by using TRAIN & TEST SPLIT FUNCTION. 
# After that, train with training set & test with testing set.

# In[10]:


# Import packages
from sklearn.model_selection import train_test_split


# In[11]:


# TRAIN & TEST SPLIT in 4, with 20% in testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Print the results
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ### Classification
# ### K nearest neighbor (KNN)

# In[12]:


# Import packages
from sklearn.neighbors import KNeighborsClassifier


# ##### Training

# In[13]:


# Train starting the algorithm with k=4
k = 4

# Train/Fit the Model
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# ##### Predicting

# In[14]:


# Predict
yhat = neigh.predict(X_test)
yhat[0:5]


# ### Accuracy evaluation
# It calculates how closely the actual labels and predicted labels are matched in the test set.
# 
# This function is equal to the jaccard_similarity_score function. 

# In[17]:


# Import packages
from sklearn import metrics

# Print the values of Accuracy for Train set & Test set
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ##### Practice with k = 6

# In[18]:


# Define k = 6
k = 6

# Train/Fit the model
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

# Predict
yhat6 = neigh6.predict(X_test)

# Print the values of Accuracy for Train set & Test set
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))


# In[19]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[20]:


plt.plot(range(1,Ks),mean_acc,'r')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.70)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[21]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




