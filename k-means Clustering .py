#!/usr/bin/env python
# coding: utf-8

# # k-means Clustering

# In[1]:


# Import packages
import random # library for random number generation
import numpy as np # library for vectorized computation
import pandas as pd # library to process data as dataframes

import matplotlib.pyplot as plt # plotting library
# backend for rendering plots within the browser
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs

print('Libraries imported.')


# ### 1. k-means on a Randomly Generated Dataset
# 
# How k-means works with an example of engineered datapoints
# 
# 30 data points belonging to 2 different clusters (x1 is the first feature and x2 is the second feature)

# In[2]:


# Data
x1 = [-4.9, -3.5, 0, -4.5, -3, -1, -1.2, -4.5, -1.5, -4.5, -1, -2, -2.5, -2, -1.5, 4, 1.8, 2, 2.5, 3, 4, 2.25, 1, 0, 1, 2.5, 5, 2.8, 2, 2]
x2 = [-3.5, -4, -3.5, -3, -2.9, -3, -2.6, -2.1, 0, -0.5, -0.8, -0.8, -1.5, -1.75, -1.75, 0, 0.8, 0.9, 1, 1, 1, 1.75, 2, 2.5, 2.5, 2.5, 2.5, 3, 6, 6.5]

print('Datapoints defined!')


# ##### Define a function that assigns each datapoint to a cluster

# In[3]:


colors_map = np.array(['b', 'r'])
def assign_members(x1, x2, centers):
    compare_to_first_center = np.sqrt(np.square(np.array(x1) - centers[0][0]) + np.square(np.array(x2) - centers[0][1]))
    compare_to_second_center = np.sqrt(np.square(np.array(x1) - centers[1][0]) + np.square(np.array(x2) - centers[1][1]))
    class_of_points = compare_to_first_center > compare_to_second_center
    colors = colors_map[class_of_points + 1 - 1]
    return colors, class_of_points

print('assign_members function defined!')


# ##### Define a function that updates the centroid of each cluster

# In[4]:


# update means
def update_centers(x1, x2, class_of_points):
    center1 = [np.mean(np.array(x1)[~class_of_points]), np.mean(np.array(x2)[~class_of_points])]
    center2 = [np.mean(np.array(x1)[class_of_points]), np.mean(np.array(x2)[class_of_points])]
    return [center1, center2]

print('assign_members function defined!')


# ##### Define a function that plots the data points along with the cluster centroids

# In[5]:


def plot_points(centroids=None, colors='g', figure_title=None):
    # plot the figure
    fig = plt.figure(figsize=(15, 10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)
    
    centroid_colors = ['bx', 'rx']
    if centroids:
        for (i, centroid) in enumerate(centroids):
            ax.plot(centroid[0], centroid[1], centroid_colors[i], markeredgewidth=5, markersize=20)
    plt.scatter(x1, x2, s=500, c=colors)
    
    # define the ticks
    xticks = np.linspace(-6, 8, 15, endpoint=True)
    yticks = np.linspace(-6, 6, 13, endpoint=True)

    # fix the horizontal axis
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # add tick labels
    xlabels = xticks
    ax.set_xticklabels(xlabels)
    ylabels = yticks
    ax.set_yticklabels(ylabels)

    # style the ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params('both', length=2, width=1, which='major', labelsize=15)
    
    # add labels to axes
    ax.set_xlabel('x1', fontsize=20)
    ax.set_ylabel('x2', fontsize=20)
    
    # add title to figure
    ax.set_title(figure_title, fontsize=24)

    plt.show()

print('plot_points function defined!')


# ##### Initialize k-means - plot data points

# In[6]:


# Plot data points
plot_points(figure_title='Scatter Plot of x2 vs x1')


# ##### Initialize k-means - randomly define clusters and add them to plot

# In[7]:


# Creating centers/k-means
centers = [[-2, 2], [2, -2]]

# Plotting centers/k-means into a figure
plot_points(centers, figure_title='k-means Initialization')


# ##### Run k-means (4-iterations only)

# In[8]:


number_of_iterations = 4
for i in range(number_of_iterations):
    input('Iteration {} - Press Enter to update the members of each cluster'.format(i + 1))
    colors, class_of_points = assign_members(x1, x2, centers)
    title = 'Iteration {} - Cluster Assignment'.format(i + 1)
    plot_points(centers, colors, figure_title=title)
    input('Iteration {} - Press Enter to update the centers'.format(i + 1))
    centers = update_centers(x1, x2, class_of_points)
    title = 'Iteration {} - Centroid Update'.format(i + 1)
    plot_points(centers, colors, figure_title=title)


# #### Generating Random Data 

# In[9]:


# Set up a random seed to 0
np.random.seed(0)


# #### Generating Random Clusters
# 
# *Random Clusters* of points by using the **make_blobs** class
# 
# 
# <b> <u> Input </u> </b>
# <ul>
#     <li> <b>n_samples</b>: The total number of points equally divided among clusters. </li>
#     <ul> <li> Value will be: 5000 </li> </ul>
#     <li> <b>centers</b>: The number of centers to generate, or the fixed center locations. </li>
#     <ul> <li> Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]] </li> </ul>
#     <li> <b>cluster_std</b>: The standard deviation of the clusters. </li>
#     <ul> <li> Value will be: 0.9 </li> </ul>
# </ul>

# In[10]:


# Creating random clusters
X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)


# ##### Setting up k-means
# 
# KMeans class has many parameters but we will use these three:
# <ul>
# 
# <li> <strong>init</strong>: Initialization method of the centroids. </li>
#     <ul>
#         <li> Value will be: "k-means++". k-means++ selects initial cluster centers for <em>k</em>-means clustering in a smart way to speed up convergence.</li>
#     </ul>
#     <li> <strong>n_clusters</strong>: The number of clusters to form as well as the number of centroids to generate. </li>
#     <ul> <li> Value will be: 4 (since we have 4 centers)</li> </ul>
#     <li> <strong>n_init</strong>: Number of times the <em>k</em>-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. </li>
#     <ul> <li> Value will be: 12 </li> </ul>
# </ul>
# 
# Initialize KMeans with these parameters, where the output parameter is called **k_means**.

# In[11]:


# Set k-means
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)


# ##### Fitting the k-means model on X

# In[12]:


# Fit k-means on X
k_means.fit(X)


# In[13]:


# Grab the labels for each point in the model 
k_means_labels = k_means.labels_
k_means_labels


# In[14]:


# Get the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# ##### Visualizing the Resulting Clusters

# In[16]:


# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(15, 10))

# colors uses a color map, which will produce an array of colors based on
# the number of labels. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# create a plot
ax = fig.add_subplot(1, 1, 1)

# loop through the data and plot the datapoints and centroids.
# k will range from 0-3, which will match the number of clusters in the dataset.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # create a list of all datapoints, where the datapoitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # plot the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        
    # plot the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# title of the plot
ax.set_title('KMeans')

# remove x-axis ticks
ax.set_xticks(())

# remove y-axis ticks
ax.set_yticks(())

# show the plot
plt.show()


# ## Using k-means for Customer Segmentation

# In[18]:


# Load data
customers_df = pd.read_csv('Downloads/Cust_Segmentation.csv')
customers_df.head()


# ##### Pre-processing
# 
# The feature 'Address' is a categorical variable.
# 
# ##### Note: k-means algorithm isn't directly applicable to categorical variables because Euclidean distance function isn't really meaningful for discrete variables. So, drop feature 'Address' and run clustering.

# In[19]:


# Drop/Remove feature 'Address' 
df = customers_df.drop('Address', axis=1)
df.head()


# ##### Normalization is a statistical method that helps mathematical-based algorithms interpret features with different magnitudes and distributions equally. Use **StandardScaler()** to normalize our dataset.

# In[20]:


# Import packages
from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]
X = np.nan_to_num(X)
cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset


# #### Modeling (Modeling/Fitting/Transforming)

# In[21]:


# Group the customers into three clusters
num_clusters = 3
# Modeling/Fitting/Transforming
k_means = KMeans(init="k-means++", n_clusters=num_clusters, n_init=12)
k_means.fit(cluster_dataset)
labels = k_means.labels_

print(labels)


# #### Exploring the results
# 
# Note: each row in our dataset represents a customer and therefore, each row is assigned a label.

# In[23]:


# Showing the column 'Labels'
df["Labels"] = labels
df.head(5)


# In[24]:


# Checking the centroid values by averaging the features in each cluster
df.groupby('Labels').mean()


# In[25]:


df.columns


# In[ ]:




