#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression

# ## Difference between Linear and Logistic Regression

# <a id="ref1"></a>
# ## What is the difference between Linear and Logistic Regression?
# 
# While Linear Regression is suited for estimating continuous values (e.g. estimating house price), it is not the best tool for predicting the class of an observed data point. In order to estimate the class of a data point, we need some sort of guidance on what would be the <b>most probable class</b> for that data point. For this, we use <b>Logistic Regression</b>.
# 
# <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# <font size = 3><strong>Recall linear regression:</strong></font>
# <br>
# <br>
#     As you know, <b>Linear regression</b> finds a function that relates a continuous dependent variable, <b>y</b>, to some predictors (independent variables $x_1$, $x_2$, etc.). For example, Simple linear regression assumes a function of the form:
# <br><br>
# $$
# y = \theta_0 + \theta_1  x_1 + \theta_2  x_2 + \cdots
# $$
# <br>
# and finds the values of parameters $\theta_0, \theta_1, \theta_2$, etc, where the term $\theta_0$ is the "intercept". It can be generally shown as:
# <br><br>
# $$
# ℎ_\theta(𝑥) = \theta^TX
# $$
# <p></p>
# 
# </div>
# 
# Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, <b>y</b>, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.
# 
# Logistic regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into a probability with the following function, which is called sigmoid function 𝜎:
# 
# $$
# ℎ_\theta(𝑥) = \sigma({\theta^TX}) =  \frac {e^{(\theta_0 + \theta_1  x_1 + \theta_2  x_2 +...)}}{1 + e^{(\theta_0 + \theta_1  x_1 + \theta_2  x_2 +\cdots)}}
# $$
# Or:
# $$
# ProbabilityOfaClass_1 =  P(Y=1|X) = \sigma({\theta^TX}) = \frac{e^{\theta^TX}}{1+e^{\theta^TX}} 
# $$
# 
# In this equation, ${\theta^TX}$ is the regression result (the sum of the variables weighted by the coefficients), `exp` is the exponential function and $\sigma(\theta^TX)$ is the sigmoid or [logistic function](http://en.wikipedia.org/wiki/Logistic_function), also called logistic curve. It is a common "S" shape (sigmoid curve).
# 
# So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability:
# 
# <img
# src="https://ibm.box.com/shared/static/kgv9alcghmjcv97op4d6onkyxevk23b1.png" width="400" align="center">
# 
# 
# The objective of __Logistic Regression__ algorithm, is to find the best parameters θ, for $ℎ_\theta(𝑥)$ = $\sigma({\theta^TX})$, in such a way that the model best predicts the class of each case.

# #### Customer churn with Logistic Regression

# In[2]:


# Import packages
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


# Load data
churn_df = pd.read_csv("Downloads/ChurnData.csv")
churn_df.head()


# In[4]:


# Shape datatset
churn_df.shape


# In[5]:


# Column names
churn_df.columns


# ##### Data pre-processing and selection

# In[6]:


# Selecting features for the modeling
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

# Change the target data type to be integer (as it is a requirement by the skitlearn algorithm)
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()


# In[7]:


# Define X, transforming the pandas df in to a np array (asarray function)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[8]:


# Define y, transforming the pandas df in to a np array (asarray function)
y = np.asarray(churn_df['churn'])
y [0:5]


# In[9]:


# Import packages
from sklearn import preprocessing

# Normalize/Fit/Transform the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ##### Train & Test Split dataset

# In[10]:


# Import packages
from sklearn.model_selection import train_test_split


# In[11]:


# Train & Test Split in 4(X_train, X_test, y_train, y_test) with 20% in testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Print splitting's results 
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# #### Modeling 

# In[12]:


# Import packages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[13]:


# Modeling 
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# #### Predict

# In[14]:


# Predicting LR (LogisticRegression of X/y_train) in function to X test(features/indipendent)
yhat = LR.predict(X_test)
yhat


# ##### predict_proba function: 
# returns estimates for all classes, ordered by the label of classes. 
# 
# So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X)

# In[15]:


# Predicting probabilities in function to X test(features/all classes/indipendent)
yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[16]:


# Higher probability
yhat_prob.max()


# In[17]:


# Lower probability
yhat_prob.min()


# #### Evaluation

# ##### JACCARD INDEX accuracy evaluation:
# It's the size of the intersection divided by the size of the union of two label sets. 
# 
# If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

# In[18]:


# Import packages
from sklearn.metrics import jaccard_similarity_score


# In[19]:


# JACCARD INDEX accuracy evaluation
jaccard_similarity_score(y_test, yhat)


# ##### CONFUSION MATRIX accuracy evaluation:
# Cheking the accuracy of classifier

# In[20]:


# Import packages
from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[22]:


# Define plot confusion_matrix function
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
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[ ]:





# In[ ]:





# In[ ]:




