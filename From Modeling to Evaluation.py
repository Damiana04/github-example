#!/usr/bin/env python
# coding: utf-8

# #### From Modeling to Evaluation

# In[1]:


# Import packages
import pandas as pd # import library to read data into dataframe
pd.set_option("display.max_columns", None)
import numpy as np # import numpy library
import re # import library for regular expression
import random # library for random number generation


# In[2]:


# LOad data
recipes = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DS0103EN/labs/data/recipes.csv")
recipes.head()


# In[3]:


# Fix name of the column displaying the cuisine
column_names = recipes.columns.values
column_names[0] = "cuisine"
recipes.columns = column_names
# Ensuring the result
recipes.head()


# In[4]:


# Convert cuisine names to lower case
recipes["cuisine"] = recipes["cuisine"].str.lower()
recipes.head()


# In[5]:


# Make the cuisine names consistent
recipes.loc[recipes["cuisine"] == "austria", "cuisine"] = "austrian"
recipes.loc[recipes["cuisine"] == "belgium", "cuisine"] = "belgian"
recipes.loc[recipes["cuisine"] == "china", "cuisine"] = "chinese"
recipes.loc[recipes["cuisine"] == "canada", "cuisine"] = "canadian"
recipes.loc[recipes["cuisine"] == "netherlands", "cuisine"] = "dutch"
recipes.loc[recipes["cuisine"] == "france", "cuisine"] = "french"
recipes.loc[recipes["cuisine"] == "germany", "cuisine"] = "german"
recipes.loc[recipes["cuisine"] == "india", "cuisine"] = "indian"
recipes.loc[recipes["cuisine"] == "indonesia", "cuisine"] = "indonesian"
recipes.loc[recipes["cuisine"] == "iran", "cuisine"] = "iranian"
recipes.loc[recipes["cuisine"] == "italy", "cuisine"] = "italian"
recipes.loc[recipes["cuisine"] == "japan", "cuisine"] = "japanese"
recipes.loc[recipes["cuisine"] == "israel", "cuisine"] = "jewish"
recipes.loc[recipes["cuisine"] == "korea", "cuisine"] = "korean"
recipes.loc[recipes["cuisine"] == "lebanon", "cuisine"] = "lebanese"
recipes.loc[recipes["cuisine"] == "malaysia", "cuisine"] = "malaysian"
recipes.loc[recipes["cuisine"] == "mexico", "cuisine"] = "mexican"
recipes.loc[recipes["cuisine"] == "pakistan", "cuisine"] = "pakistani"
recipes.loc[recipes["cuisine"] == "philippines", "cuisine"] = "philippine"
recipes.loc[recipes["cuisine"] == "scandinavia", "cuisine"] = "scandinavian"
recipes.loc[recipes["cuisine"] == "spain", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "portugal", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "switzerland", "cuisine"] = "swiss"
recipes.loc[recipes["cuisine"] == "thailand", "cuisine"] = "thai"
recipes.loc[recipes["cuisine"] == "turkey", "cuisine"] = "turkish"
recipes.loc[recipes["cuisine"] == "vietnam", "cuisine"] = "vietnamese"
recipes.loc[recipes["cuisine"] == "uk-and-ireland", "cuisine"] = "uk-and-irish"
recipes.loc[recipes["cuisine"] == "irish", "cuisine"] = "uk-and-irish"

recipes.head()


# In[6]:


# Remove data for cuisines with < 50 recipes
recipes_counts = recipes["cuisine"].value_counts()
cuisine_indices =  recipes_counts > 50

# Create a df for which 'cuisine' that you want to keep
cuisine_to_keep = list(np.array(recipes_counts.index.values)[np.array(cuisine_indices)])
recipes = recipes.loc[recipes["cuisine"].isin(cuisine_to_keep)]


# In[7]:


# Convert all Yes's to 1's and the No's to 0's
recipes = recipes.replace(to_replace="Yes", value=1)
recipes = recipes.replace(to_replace="No", value=0)
recipes.head()


# #### Data Modeling

# In[8]:


# import Decision T~rees scikit-learn libraries
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

get_ipython().system('conda install python-graphviz --yes')
import graphviz

from sklearn.tree import export_graphviz

import itertools


# In[9]:


# Check data again
recipes.head()


# In[10]:


# Analyzing Asian & Indian cuisines

# Select subset of cuisines
asian_indian_recipes = recipes[recipes.cuisine.isin(["korean", "japanese", "chinese", "thai", "indian"])]

# Define X,y: 'ingredients' (feature/indipendent/X) & 'cusines' (target/dependent variable/y)
cuisines = asian_indian_recipes["cuisine"]
ingredients = asian_indian_recipes.iloc[:,1:]

# Create an instance of Decision Tree Model
bamboo_tree = tree.DecisionTreeClassifier(max_depth=3)
# Fit the Decision Tree model 
bamboo_tree.fit(ingredients, cuisines)

print("Decision tree model saved to bamboo_tree!")


# In[11]:


# Plot the decision tree and examine it
export_graphviz(bamboo_tree,
                feature_names=list(ingredients.columns.values),
                out_file="bamboo_tree.dot",
                class_names=np.unique(cuisines),
                filled=True,
                node_ids=True,
                special_characters=True,
                impurity=False,
                label="all",
                leaves_parallel=False)

with open("bamboo_tree.dot") as bamboo_tree_image:
    bamboo_tree_graph = bamboo_tree_image.read()
graphviz.Source(bamboo_tree_graph)


# The decision tree learned:
# * If a recipe contains *cumin* and *fish* and **no** *yoghurt*, then it is most likely a **Thai** recipe.
# * If a recipe contains *cumin* but **no** *fish* and **no** *soy_sauce*, then it is most likely an **Indian** recipe.

# #### Training, Testing & Splitting

# In[12]:


# Keeping the previous dataset as Model
bamboo = recipes[recipes.cuisine.isin(["korean", "japanese", "chinese", "thai", "indian"])]


# In[13]:


# How many recipes exist for each cuisine
bamboo["cuisine"].value_counts()


# In[14]:


# Take 30 recipes from each cuisine to use as the test set(sample) called bamboo_test

# Define the sample size
sample_n = 30


# In[15]:


# Take 30 recipes from each cuisine

# Set random seed
random.seed(1234)

# Define the test set
bamboo_test = bamboo.groupby("cuisine", group_keys=False).apply(lambda x: x.sample(sample_n))

# Define ingredients as X_test (feature/indipendent/X)
bamboo_test_ingredients = bamboo_test.iloc[:,1:]

# Define cuisines as y_test (target/dependent/y)
bamboo_test_cuisine = bamboo_test["cuisine"]


# In[16]:


# Check that there are 30 recipes for each cuisine
bamboo_test["cuisine"].value_counts()


# In[17]:


# Create the training set by removing the test set from the bamboo dataset, and let's call the training set bamboo_train
bamboo_test_index = bamboo.index.isin(bamboo_test.index)

# Define the train set
bamboo_train = bamboo[~bamboo_test_index]

# Define ingredients as X_train (feature/indipendent/X)
bamboo_train_ingredients = bamboo_train.iloc[:,1:]

# Define cuisines as y_train (target/dependent/y)
bamboo_train_cuisine = bamboo_train["cuisine"]


# In[18]:


# Check that there are 30 fewer recipes now for each cuisine
bamboo_train["cuisine"].value_counts()


# ##### Prediction

# In[19]:


# Create an instance (DecisionTree object)
bamboo_train_tree = tree.DecisionTreeClassifier(max_depth=15)

# Fit the train set
bamboo_train_tree.fit(bamboo_train_ingredients, bamboo_train_cuisine)


# In[20]:


export_graphviz(bamboo_train_tree,
                feature_names=list(bamboo_train_ingredients.columns.values),
                out_file="bamboo_train_tree.dot",
                class_names=np.unique(bamboo_train_cuisine),
                filled=True,
                node_ids=True,
                special_characters=True,
                impurity=False,
                label="all",
                leaves_parallel=False)

with open("bamboo_train_tree.dot") as bamboo_train_tree_image:
    bamboo_train_tree_graph = bamboo_train_tree_image.read()
graphviz.Source(bamboo_train_tree_graph)


# In[21]:


# Predict in function to bamboo_test (X/features/indipendent)
bamboo_pred_cuisines = bamboo_train_tree.predict(bamboo_test_ingredients)


# In[22]:


# Print Prediction's result
print(bamboo_pred_cuisines[0:5])


# In[23]:


# Print test's result
print(bamboo_test_ingredients[0:5])


# ##### Confusion matrix: 
# Understanding how well the decision tree is able to determine the cuisine of each recipe correctly. 
# It presents a summary on how many recipes from each cuisine are correctly classified. It also sheds some light on what cuisines are being confused with what other cuisines.

# In[24]:


# Confusion matrix

test_cuisines = np.unique(bamboo_test_cuisines)
bamboo_confusion_matrix = confusion_matrix(bamboo_test_cuisines, bamboo_pred_cuisines, test_cuisines)
title = 'Bamboo Confusion Matrix'
cmap = plt.cm.Blues

plt.figure(figsize=(8, 6))
bamboo_confusion_matrix = (
    bamboo_confusion_matrix.astype('float') / bamboo_confusion_matrix.sum(axis=1)[:, np.newaxis]
    ) * 100

plt.imshow(bamboo_confusion_matrix, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(test_cuisines))
plt.xticks(tick_marks, test_cuisines)
plt.yticks(tick_marks, test_cuisines)

fmt = '.2f'
thresh = bamboo_confusion_matrix.max() / 2.
for i, j in itertools.product(range(bamboo_confusion_matrix.shape[0]), range(bamboo_confusion_matrix.shape[1])):
    plt.text(j, i, format(bamboo_confusion_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if bamboo_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()


# In[ ]:


recipes.find("almond")


# In[ ]:




