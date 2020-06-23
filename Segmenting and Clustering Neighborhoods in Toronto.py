#!/usr/bin/env python
# coding: utf-8

# ## Toronto

# In[11]:


import pandas as pd
import numpy as np
get_ipython().system('pip install beautifulsoup4')
from bs4 import BeautifulSoup
import requests

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize  # tranform JSON file into a pandas dataframe

import folium # map rendering library

# import k-means from clustering stage
from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[12]:


# Url
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'


# In[13]:


# Sending the request
request = requests.get(url).text
soup = BeautifulSoup(request, 'lxml')
request


# In[14]:


# Sending the request
source = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
soup = BeautifulSoup(source, 'lxml')

# Setting up the table
table = soup.find("table")
table_rows = table.find_all("tr")

# Filling the table by loop
data = []
for tr in table_rows:
    td = tr.find_all("td")
    row = [tr.text for tr in td]
    
    # Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.
    if row != [] and row[1] != "Not assigned\n":
        # If a cell has a borough but a "Not assigned" neighborhood, then the neighborhood will be the same as the borough.
        if "Not assigned\n" in row[2]: 
            row[2] = row[1]
        data.append(row)

# -Dataframe with 3 columns
df_1 = pd.DataFrame(data, columns = ["PostalCode", "Borough", "Neighborhood"])
df_1.head()


# In[15]:


df_1["Neighborhood"] = df_1["Neighborhood"].str.replace("\n","")
df_1.head()


# In[16]:


df_1["PostalCode"] = df_1["PostalCode"].str.replace("\n","")
df_1.head() 


# In[17]:


df_1["Borough"] = df_1["Borough"].str.replace("\n","")
df_1.head() 


# In[82]:


# Counting unique postalcodes
postalcodes = df_1['PostalCode'].nunique()
boroughs = df_1['Borough'].nunique()
neighborhoods= df_1['Neighborhood'].nunique()
print('Unique Postalcodes : ' + str(postalcodes))
print('Unique Boroughs  : '+ str(boroughs))
print('Unique Neighborhoods  :' + str(neighborhoods))


# In[18]:


# Group all neighborhoods with the same postal code
df_1 = df_1.groupby(["PostalCode", "Borough"])["Neighborhood"].apply(", ".join).reset_index()
df_1.head()


# In[19]:


df_1.shape


# ##### Loading the second dataset

# In[20]:


# Load data 
df_2 = pd.read_csv("Downloads/Geospatial_Coordinates.csv")
df_2.head()


# ##### Merge the first dataset with the second one, creating a new dataframe

# In[21]:


# Creating a new dataframe joing the first two
df_geo = pd.merge(df_1, df_2, how='left', left_on = 'PostalCode', right_on = 'Postal Code')
# Remove the "Postal Code" column
df_geo.drop("Postal Code", axis=1, inplace=True)
df_geo.head()


# In[22]:


# Column names
df_geo.columns


# In[83]:


# Shape of the new dataframe
df_geo.shape


# In[23]:


# Get the latitude & longitude of Toronto
address = 'Toronto, ON'

# Define a user_agent called my_explorer
geolocator = Nominatim(user_agent="my_explorer")
# Define an instance of the decoder
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print("The geographical coordinates of Toronto are {}, {}.".format(latitude, longitude))


# In[24]:


# Creating a map of Toronto
map_toronto = folium.Map(location=[latitude,longitude], zoom_start=10)
map_toronto


# In[25]:


# Defining the values
neighborhoods_data = df_geo
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude']
neighborhoods = pd.DataFrame(columns=column_names)


# In[26]:


# Add markers to map
for lat, lng, Borough, Neighborhood in zip(neighborhoods_data["Latitude"], neighborhoods_data["Longitude"], neighborhoods_data["Borough"], neighborhoods_data["Neighborhood"]):
    label = '{}', '{}'.format(Neighborhood, Borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
    [lat, lng],
    radius=5,
    popup=label,
    color='green',
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.7,
    parse_html=False).add_to(map_toronto)
    
map_toronto


# In[27]:


# Credentials
CLIENT_ID = 'QESD2LMCCD405QTTBGSGP54YPJW03YV1OWTL15VM33VENKJE'
CLIENT_SECRET = 'SCLWB1SGVSMVXUEATLEMHVVB1LMST3INIQEHBOM4MVNOYKDX'
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[32]:


# Let's create a function to repeat the same process to all the neighborhoods in Toronto
LIMIT = 100
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[33]:


neighborhoods_venues = getNearbyVenues(names=neighborhoods_data['Neighborhood'],
                                   latitudes=neighborhoods_data['Latitude'],
                                   longitudes=neighborhoods_data['Longitude']
                                  )


# In[34]:


neighborhoods_venues.head()


# In[45]:


neighborhoods_venues.groupby('Venue Category').count()


# In[37]:


# See which values are present in a particular column
neighborhoods_venues[('Venue Category')].value_counts().idxmax()


# In[38]:


neighborhoods_venues[neighborhoods_venues['Venue Category']== "Yoga Studio"].count()


# In[46]:


print('There are {} uniques categories.'.format(len(neighborhoods_venues['Venue Category'].unique())))


# In[48]:


# one hot encoding
toronto_onehot = pd.get_dummies(neighborhoods_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = neighborhoods_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[49]:


toronto_onehot.shape


# In[50]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[51]:


toronto_grouped.shape


# In[52]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[53]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[56]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[57]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[77]:


# Get the latitude & longitude of Toronto
address = 'Toronto, ON'

# Define a user_agent called my_explorer
geolocator = Nominatim(user_agent="my_explorer")
# Define an instance of the decoder
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print("The geographical coordinates of Toronto are {}, {}.".format(latitude, longitude))


# In[ ]:





# In[ ]:





# In[ ]:




