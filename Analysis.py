#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.style.use('ggplot')


# In[34]:


csv_path = 'Topic_Survey_Assignment.csv'
df_sr = pd.read_csv('Downloads\Topic_Survey_Assignment.csv', index_col = 0)
df_sr.head()


# In[35]:


df.columns


# In[36]:


df_sr.max()


# In[37]:


df_sr.mean()


# In[28]:


# Sorting the values
df_sr.sort_values(['Very interested'], ascending=False, axis=0, inplace=True)

# Taking the percentage of the responses and rounding it to 2 decimal places 
df_sr = round((df_sr/2233) *100,2)

# View top 5 rows of the data 
df_sr.head()


# In[43]:


# Plotting
ax = df_sr.plot(kind='bar', 
                figsize=(20, 8),
                rot=90,color = ['#5cb85c','#5bc0de','#d9534f'],
                width=.8,fontsize=14)

# Setting plot title
ax.set_title('Percentage of Respondents Interest in Data Science Areas',fontsize=16)

# Setting figure background color
ax.set_facecolor('white')

# setting legend font size
ax.legend(fontsize=14,facecolor = 'white') 

# Removing the Border 
ax.get_yaxis().set_visible(False)

# Creating a function to display the percentage.
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )
plt.show()


# In[44]:


# Read in the data set
df_sfc = pd.read_csv('https://ibm.box.com/shared/static/nmcltjmocdi8sd5tk93uembzdec8zyaq.csv')

print('Dataset downloaded and read into a pandas dataframe!')
df_sfc.head()

# Assigning a variable with the total counts of each Neighborhood
df_neig= df_sfc['PdDistrict'].value_counts()

# Assigning the values of the variable to a Pandas Data frame
df_neig1 = pd.DataFrame(data=df_neig.values, index = df_neig.index, columns=['Count'])

# Reindexing the data frame to the requirement
df_neig1 = df_neig1.reindex(["CENTRAL", "NORTHERN", "PARK", "SOUTHERN", "MISSION", "TENDERLOIN", "RICHMOND", "TARAVAL", "INGLESIDE", "BAYVIEW"])

# Resetting the index
df_neig1 = df_neig1.reset_index()

# Assignming the column names
df_neig1.rename({'index': 'Neighborhood'}, axis='columns', inplace=True)

# View the data frame
df_neig1


# In[47]:


# Load the packages for creating the Choropleth map
get_ipython().system('pip install folium')
import folium

# Read in the GeoJSON file
geojson = r'https://cocl.us/sanfran_geojson'

# Create the map centering San Fransico
sf_map = folium.Map(location = [37.77, -122.42], zoom_start = 12)

# Display the map
sf_map.choropleth(geo_data=geojson,
                  data=df_neig1,
                  columns=['Neighborhood', 'Count'],
                  key_on='feature.properties.DISTRICT',
                  fill_color='YlOrRd', 
                  fill_opacity=0.7, 
                  line_opacity=0.2,
                  legend_name='Crime Rate in San Francisco'
)

sf_map


# In[ ]:




