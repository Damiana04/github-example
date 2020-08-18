#!/usr/bin/env python
# coding: utf-8

# ### From Understanding to Preparation

# In[1]:


# Import packages
import pandas as pd 
pd.set_option('display.max_columns', None)
import numpy as np 
import re # import library for regular expression


# In[2]:


# Loading data
recipes = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DS0103EN/labs/data/recipes.csv")

print("Data read into dataframe!")


# In[4]:


# Look at the data
recipes.head()


# In[5]:


# Shape data
recipes.shape


# In[14]:


# Seaching if there are rice, wasaby & soy in the dataset

# Define the instance for ingredients: 
#   the 'ingredients' which I'll ask for = are a 'list'(into the df 'recipes'.between columns.give me back the values)
#   Location:
#   from df 'recipes'
#   between columns
#   between values
ingredients = list(recipes.columns.values)

# Create a Loop for each ingradients that yor're looking for:
#   Pythy, find the MATCH & GROUP it. Do it FOR input(ingredients0 in ingredients(list/where), do it FOR that MATCH above(input: match.group(0)) IN the REport COMPILEd presenting the word 'rice', SEARCH(ing) into the list 'ingredients. So, IF there's match, just PRINT it down the ingredients.
print([match.group(0) for ingredients in ingredients for match in [(re.compile(".*(rice).*")).search(ingredients)] if match])
# Just do it again for wasabi
print([match.group(0) for ingredients in ingredients for match in [(re.compile(".*(wasabi).*")).search(ingredients)] if match])
# Once again for the thirt ingedient soy
print([match.group(0) for ingredients in ingredients for match in [(re.compile(".*(soy).*")).search(ingredients)] if match])


# In[16]:


# Counting values by Countries
recipes["country"].value_counts()


# In[18]:


# Change the column name
column_names = recipes.columns.values
column_names[0] = "cuisine"
recipes.columns = column_names


# In[21]:


# Check the result
recipes.head()


# In[22]:


# Make all the cuisine names lowercase
recipes["cuisine"] = recipes["cuisine"].str.lower()


# In[23]:


# Check the result
recipes


# In[25]:


# Changing the cuisine names
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

# Check the result
recipes


# In[103]:


# Count again the values in cuisine before then change it
recipes["cuisine"].value_counts()


# In[104]:


# Remove cuisines with < 50 recipes
recipes_counts = recipes["cuisine"].value_counts()
cuisines_indices = recipes_counts > 50

cuisines_to_keep = list(np.array(recipes_counts.index.values)[np.array(cuisines_indices)])
cuisines_to_keep


# In[105]:


english = recipes[recipes["cuisine"] == "english_scottish"]
english


# In[106]:


# Number of rows of original dataframe
rows_before = recipes.shape[0]
print("Number of rows of original dataframe is {}:".format(rows_before))

recipes = recipes.loc[recipes["cuisine"].isin(cuisines_to_keep)]

# Number of rows after
rows_after = recipes.shape[0]
print("Number of rows of original dataframe is {}:".format(rows_after))

# Print the result
print("{} total rows removed".format(rows_before - rows_after))


# In[107]:


# Convert all Yes's to 1's and the No's to 0's
recipes = recipes.replace(to_replace="Yes", value=1)
recipes = recipes.replace(to_replace="No", value=0)
recipes.head()


# In[108]:


# LOOP to get the recipes that contain 'rice' and 'soy' and 'wasabi' and 'seaweed'

check_recipes = recipes.loc[
    (recipes["rice"] == 1 ) &
    (recipes["soy_sauce"] == 1) &
    (recipes["wasabi"] == 1) &
    (recipes["seaweed"] ==1)
]


# Check the results
check_recipes


# In[109]:


# Check how many recipes for each 'cuisine'
check_recipes["cuisine"].value_counts()


# In[110]:


# Count the ingredients across all recipes
ing = recipes.iloc[:, 1:].sum(axis=0)
ing


# In[111]:


# Define each column as a pandas series
ingredient = pd.Series(ing.index.values, index = np.arange(len(ing)))
count = pd.Series(list(ing), index = np.arange(len(ing)))

# Create the dataframe
ing_df = pd.DataFrame(dict(ingredient = ingredient, count = count))
ing_df = ing_df[["ingredient", "count"]]
print(ing_df.to_string())


# In[112]:


# Sort the dataframe in descending order
ing_df.sort_values(["count"], ascending=False, inplace=True)
ing_df


# In[113]:


# Create a profile for each cuisine
cuisines = recipes.groupby("cuisine").mean()
cuisines.head()


# In[114]:


# Create a profile for each cuisine by displaying the top four ingredients in each cuisine

num_ingredients = 10 # define number of top ingredients to print

# Define number of top ingredients to print
def print_top_ingredients(row):
    print(row.name.upper())
    row_sorted = row.sort_values(ascending=False)*100
    top_ingredients = list(row_sorted.index.values)[0:num_ingredients]
    row_sorted = list(row_sorted)[0:num_ingredients]

    for ind, ingredient in enumerate(top_ingredients):
        print("%s (%d%%)" % (ingredient, row_sorted[ind]), end=' ')
    print("\n")


# Apply function to cuisines dataframe
create_cuisines_profiles = cuisines.apply(print_top_ingredients, axis=1)


# In[ ]:





# In[ ]:




