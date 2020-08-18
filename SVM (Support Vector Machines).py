#!/usr/bin/env python
# coding: utf-8

# ## SVM 

# In[1]:


# Import packages
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# Loading data
cell_df = pd.read_csv("Downloads/cell_samples.csv")
cell_df.head()


# ##### Distribution of the classes based on Clump thickness and Uniformity of cell size

# In[3]:


cell_df.shape


# In[4]:


ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()


# #### Data pre-processing and selection

# In[5]:


# Check the columns data types
cell_df.dtypes


# ##### BareNuc column includes some values that are not numerical. Drop those rows

# In[6]:


# Dropping the rows which are not numerical
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]

# Transforming categorical values to numerical for "BareNuc" feature
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

# Checking the result of the tranformation
cell_df.dtypes


# In[7]:


# Selecting features for the modeling
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]


# In[8]:


# Define X, transforming the pandas df in to a np array (asarray function)
X = np.asarray(feature_df)
X[0:5]


# ##### Class' values are float. Transform them as integer for obtaining a fiel categorized with just two values (benign (=2) or malignant (=4))

# In[9]:


# Transforming float values to integer for "Class" feature
cell_df['Class'] = cell_df['Class'].astype('int')


# In[10]:


# Define y, transforming the pandas df in to a np array (asarray function)
y = np.asarray(cell_df['Class'])
y[0:5]


# ##### Train & Test Split dataset

# In[11]:


# Train & Test Split in 4(X_train, X_test, y_train, y_test) with 20% in testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Print splitting's results 
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ##### Modeling (Modeling/Fitting/Transforming)
# 
# ##### KERNELLING: mathematical function used for the transformation. It's used for mapping data into a higher dimensional space. It can be of different types, such as:
# 
#     1.Linear
#     2.Polynomial
#     3.Radial basis function (RBF)
#     4.Sigmoid
# the default is RBF (Radial Basis Function) 

# In[12]:


# Import packages
from sklearn import svm


# In[13]:


# Modeling/Fitting/Transforming X/y_trainset by KERNEL with the default is RBF (Radial Basis Function)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# ##### Predict

# In[14]:


# Predicting SVM (Support Vector Machines of X/y_train) in function to X test(features/indipendent)
yhat = clf.predict(X_test)
yhat [0:5]


# ##### Evaluation & Visualizzation

# In[15]:


# Import packages
from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[16]:


# Define the plot function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plt.show


# In[ ]:




