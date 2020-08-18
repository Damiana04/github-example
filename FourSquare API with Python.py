#!/usr/bin/env python
# coding: utf-8

# #### FourSquare API with Python

# In[6]:


import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# In[7]:


# Define Foursquare Credentials and Version
CLIENT_ID = 'QESD2LMCCD405QTTBGSGP54YPJW03YV1OWTL15VM33VENKJE' # your Foursquare ID
CLIENT_SECRET = 'SCLWB1SGVSMVXUEATLEMHVVB1LMST3INIQEHBOM4MVNOYKDX' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# #### Example: converting the Contrad Hotel's address to its latitude and longitude coordinates
# In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent foursquare_agent

# In[8]:


# Define the Contrad Hotel's address
address = '102 North End Ave, New York, NY'

# Define the user_agent called "foursquare_agent"
geolocator = Nominatim(user_agent="foursquare_agent")

# Find out the location by the geolocator (it'll translate the address)
location = geolocator.geocode(address)

# Find out latitude & longitude
latitude = location.latitude
longitude = location.longitude

# Print out latitude & longitude' results
print(latitude, longitude)


# ##### Search for a specific venue category:
# > `https://api.foursquare.com/v2/venues/`**search**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&ll=`**LATITUDE**`,`**LONGITUDE**`&v=`**VERSION**`&query=`**QUERY**`&radius=`**RADIUS**`&limit=`**LIMIT**

# In[9]:


# Define a query to search for Italian food that is within 500 metres from the Conrad Hotel
search_query = 'Italian'
radius = 500
print(search_query + ' .... OK!')


# In[10]:


url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
url


# #### Send the GET Request and examine the results

# In[11]:


# Sending the request
results = requests.get(url).json()

# Examining the results in JSON language
results


# #### TRANSLATE JSON's RESULTS into PANDAS' DF

# In[12]:


# Assign relevant part of JSON to venues
venues = results['response']['venues']


# Tranforming Json venues into a Pandas dataframe
dataframe = json_normalize(venues)
dataframe.head()


# #### Define information of interest and filter dataframe

# In[13]:


# Keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# Function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# Filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# Clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered


# ##### Visualize the Italian restaurants that are nearby

# In[14]:


# Just the name of Italian restaurant near the hotel
dataframe_filtered.name


# ##### Visualizzation by folium: Conrad Hotel & Italian restaurants

# In[15]:


# Generate map centred around the Conrad Hotel
venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) 

# Add a red circle marker to represent the Conrad Hotel
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Conrad Hotel',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# Add the Italian restaurants as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# Display map
venues_map


# ### Explore a Given Venue
# > `https://api.foursquare.com/v2/venues/`**VENUE_ID**`?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&v=`**VERSION**

# In[16]:


# Explore the closest Italian restaurant: Harry's Italian Pizza Bar
venue_id = '4fa862b3e4b0ebff2f749f06' # ID of Harry's Italian Pizza Bar
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)
url


# ##### Send GET request for result

# In[17]:


# Sending the request
result = requests.get(url).json()

# Examine the result in JSON language
print(result['response']['venue'].keys())
result['response']['venue']


# In[18]:


# Cheking the rating
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# In[19]:


# Explore the other Italian restaurant: Conca Cucina Italian Restaurant
venue_id = '4f3232e219836c91c7bfde94' # ID of Conca Cucina Italian Restaurant
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# In[20]:


# Explore another Italian restaurant: Ecco
venue_id = '3fd66200f964a520f4e41ee3' # ID of Ecco
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# ##### Get the number of tips

# In[21]:


result['response']['venue']['tips']['count']


# ### D. Get the venue's tips
# > `https://api.foursquare.com/v2/venues/`**VENUE_ID**`/tips?client_id=`**CLIENT_ID**`&client_secret=`**CLIENT_SECRET**`&v=`**VERSION**`&limit=`**LIMIT**

# ### Create URL and send GET request

# In[22]:


## Ecco Tips
limit = 15 # set limit to be greater than or equal to the total number of tips
url = 'https://api.foursquare.com/v2/venues/{}/tips?client_id={}&client_secret={}&v={}&limit={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION, limit)

results = requests.get(url).json()
results


# #### Get tips and list of associated features

# In[25]:


tips = results['response']['tips']['items']

tips = results['response']['tips']['items'][0]
tips.keys()


# #### Format column width and display all tips

# In[26]:


pd.set_option('display.max_colwidth', -1)

tips_df = json_normalize(tips) # json normalize tips

# columns to keep
filtered_columns = ['text', 'agreeCount', 'disagreeCount', 'id', 'user.firstName', 'user.lastName', 'user.gender', 'user.id']
tips_filtered = tips_df.loc[:, filtered_columns]

# display tips
tips_filtered


# In[ ]:




