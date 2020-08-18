#!/usr/bin/env python
# coding: utf-8

# # Fashion App

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load Train dataset
fashion_train_df = pd.read_csv('Downloads/2243_9243_compressed_fashion-mnist_train.csv.zip', sep = ',')
fashion_train_df


# In[3]:


# Load Test dataset
fashion_test_df = pd.read_csv('Downloads/2243_9243_compressed_fashion-mnist_test.csv.zip', sep = ',')
fashion_test_df


# In[4]:


# Tail Train dataset
fashion_train_df.tail()


# In[5]:


# Tail Test dataset
fashion_test_df.tail()


# In[6]:


# General info of Train dataset
fashion_train_df.info


# In[7]:


# General info of Test dataset
fashion_test_df.info


# In[8]:


# Creating the new Train dataset by converting the values in floats
training = np.array(fashion_train_df, dtype ='float32')


# In[9]:


# Creating the new Test dataset by converting the values in floats
testing = np.array(fashion_test_df, dtype ='float32')


# In[10]:


# Visualizing random images

# Import random 
import random

# Define i random object
i = random.randint(1, 60000)

# Visualizing random images
plt.imshow(training[i, 1:].reshape(28, 28))

# Define labels
label = training[1, 0]
label


# In[11]:


# Labels

# 0 = T-shirt/Top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 = Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneaker
# 8 = Bag
# 9 = Ankle boot


# In[12]:


# Visualizing random images
plt.imshow(training[i, 1:].reshape(28, 28))


# In[13]:


# Checking some images
plt.imshow(training[60, 1:].reshape(28, 28))


# In[14]:


# Checking some images
plt.imshow(training[12, 1:].reshape(28, 28))


# In[15]:


# Checking some images
plt.imshow(training[10, 1:].reshape(28, 28))


# In[16]:


# Checking some images
plt.imshow(training[600, 1:].reshape(28, 28))


# In[17]:


# Let's see more images in a Grid format

# Define the dimensions of the plot Grid
W_grid = 15
L_grid = 15

# Subplot 
fig, axes = plt.subplots(L_grid, W_grid, figsize= (17,17))

# Flaten the 15 x 15 matrix into 255 array
axes = axes.ravel()

# Get the lenght of the Training dataset
n_training = len(training)

# Loop throught select a random number from 0 to n_training
# Create evenly spaces variables
for i in np.arange(0, W_grid * L_grid):
    
    # Select a random number
    index = np.random.randint(0, n_training)
    # Read & Display an image with the selected index
    axes[i].imshow(training[index, 1:].reshape(28, 28))
    axes[i].set_title(training[index, 0], fontsize=8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)


# ### TRAINING the MODEL

# In[18]:


# Define X & y in the Training dataset
X_train = training[:, 1:]/255
y_train = training[:, 0]


# In[19]:


# Define X & y in the Testing dataset
X_test = testing[:, 1:]/255
y_test = testing[:, 0]


# In[20]:


# Train, Test & Split the Model 

# Import package
from sklearn.model_selection import train_test_split

# Create the Train_Test_Split function
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size= 0.2, random_state=12345)


# In[21]:


# Reshaping for filling correctly the images
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))


# In[22]:


# Cheching the new X_train shape
X_train.shape


# In[23]:


# Cheching the new X_test shape
X_test.shape


# In[24]:


# Cheching the new X_validate shape
X_validate.shape


# In[25]:


# Import packages for the Network Neural 
get_ipython().system('pip install tensorflow')


# In[26]:


get_ipython().system('pip install keras')


# In[27]:


get_ipython().system('pip install --upgrade tensorflow')


# In[28]:


get_ipython().system('pip install keras==2.2.4')


# In[29]:


import tensorflow as tf
layers = tf.keras.layers


# In[ ]:


# Define the Sequential statement
cnn_model = Sequential()


# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(25):
 plt.subplot(5,5,i+1)
 plt.xticks([])
 plt.yticks([])
 plt.grid(False)
 plt.imshow(X_train[i], cmap=plt.cm.binary)
 plt.xlabel(class_names[y_train[i]])
plt.show()


# In[ ]:




