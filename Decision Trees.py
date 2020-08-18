#!/usr/bin/env python
# coding: utf-8

# ## Decision Trees

# In[1]:


# Import Libraries
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[2]:


# Loading data
my_data = pd.read_csv("Downloads/drug200.csv", delimiter=",")
my_data[0:5]


# In[3]:


my_data.shape


# ### Pre-processing

# In[4]:


# Define X as the Feature Matrix (data of my_data)
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# Sklearn Decision Trees do not handle categorical variables. 
# Convert these features to numerical values. pandas.get_dummies() 

# In[5]:


# Import packages
from sklearn import preprocessing

# Encoding/Translating/Reading right the labels by function preprocessing.LabelEncoding

# Encoding/Translating/Reading right the label "Sex"
le_sex = preprocessing.LabelEncoder()

# Train/Fit/Transform label "Sex"
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

# Encoding/Tranlating/Reading right the label "BP"
le_BP = preprocessing.LabelEncoder()

# Train/Fit/Transform label "BP"
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# Encoding/Translating/Reading right the label "Cholesterol"
le_Chol = preprocessing.LabelEncoder()

# Train/Fit/Transform label "Cholesterol"
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[6]:


# Define y as the response vector (target)
y = my_data["Drug"]
y[0:5]


# #### Setting up the Decision Tree

# In[7]:


# Import packages
from sklearn.model_selection import train_test_split


# Now train_test_split will return 4 different parameters. We will name them:
# X_trainset, X_testset, y_trainset, y_testset
# 
# The train_test_split will need the parameters:
# X, y, test_size=0.3, and random_state=3.
# 
# The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits.

# In[8]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# ##### Practice

# In[9]:


# Shape of X_trainset and y_trainset. Ensure that the dimensions match
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# #### Modeling: create a Decision Tree object 

# In[12]:


# Create an instance (object) of the DecisionTreeClassifier called drugTree. Classifier/criterion="entropy" 
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# ##### FIT the data with the trainset X & y

# In[13]:


# FIT the data with the TRAINING feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)


# ##### Prediction

# In[15]:


# Predicting in function to X(features/indipendent variable) with TESTset
predTree = drugTree.predict(X_testset)


# In[17]:


# Printing predTree and y_testset 
print (predTree [0:5])
print (y_testset [0:5])


# ##### Evaluation

# In[21]:


# Import packages
from sklearn import metrics
import matplotlib.pyplot as plt

# Print the Accuracy's result, calculated on y_testset & predicted value on X_testset
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# ##### Visualization

# In[19]:


# Import packages
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:




