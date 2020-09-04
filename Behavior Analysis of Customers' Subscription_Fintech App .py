#!/usr/bin/env python
# coding: utf-8

# ## Behavior Analysis of Customers' Subscription_Fintech App 

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser


# In[2]:


# Load dataset
original_df = pd.read_csv("Downloads/87836_202131_compressed_appdata10.csv.zip")
original_df.head()


# In[3]:


# Crating a copy of the original dataframe
df = original_df.copy()
df.head()


# ##### Exploratory Data Analysis

# In[4]:


# Shape
df.shape


# In[5]:


# General info 
df.info()


# In[6]:


# Statistical values
df.describe()


# In[7]:


# Checking missing values in each column
df.isnull().sum()


# In[8]:


# Columns
df.columns


# In[9]:


# Screen_list feature
df.screen_list.unique()


# In[10]:


# Checking the "Hour" feature
df.hour.tail


# In[11]:


# Cleaning the "Hour" feature to integer, using just the first two numbers
df["hour"] = df.hour.str.slice(1, 3).astype(int)


# In[12]:


# Checking the result into the "hour" feature
df["hour"].tail


# In[13]:


# Creating a second dataframe
df2 = df.copy().drop(columns = ["user", "screen_list", "enrolled_date", "first_open", "enrolled"])
df2.head()


# In[14]:


# Histograms of features
plt.figure(figsize=(12, 12))
plt.suptitle("Histogram of Numerical Columns", fontsize = 20)
for i in range(1, df2.shape[1]+ 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(df2.columns.values[i - 1])
    
    vals = np.size(df2.iloc[:, i - 1].unique())
    
    plt.hist(df2.iloc[:, i - 1], bins = vals, color = '#3F5D7D')


# ##### Correlation Matrix

# In[15]:


# Correlation Plot
df2.corrwith(df.enrolled).plot.bar(figsize = (20, 10), 
                                   title = "Correlation with Response Variable", 
                                   fontsize = 15, rot = 45, 
                                   color=['black', 'lime', 'green', 'blue', 'cyan', 'magenta', 'purple'], 
                                   edgecolor = 'yellow', 
                                   grid = True)


# In[16]:


# Correlation Matrix (background)
sns.set(style="white", font_scale=2)

# Compute the Correlation Matrix
corr = df2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and the correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[17]:


# Checking the datatypes
df.dtypes


# ##### Features Engineering

# In[18]:


# Parsing the features
df["first_open"] = [parser.parse(row_data) for row_data in df["first_open"]]
df["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data, str) else row_data for row_data in df["enrolled_date"]]


# In[19]:


# Checking the result for "first_open" feature
df["first_open"]


# In[20]:


# Checking the result for "enrolled_date" feature
df["enrolled_date"]


# In[21]:


# Checking the result
df.dtypes


# In[22]:


# Difference
df["difference"] = (df.enrolled_date - df.first_open).astype('timedelta64[h]')
df["difference"]


# In[23]:


# 'Difference' General Histogram
plt.figure(figsize=(10, 10))
plt.hist(df["difference"].dropna(), color = '#3F5D7D')
plt.title("Distribution of Time-Since-Enrolled")
plt.show()


# In[24]:


# 'Difference' Histogram in the range 0 to 100
plt.figure(figsize=(10, 10))
plt.hist(df["difference"].dropna(), color = '#3F5D7D', range= [0, 100])
plt.title("Distribution of Time-Since-Enrolled range 0-100")
plt.show()


# In[25]:


# Dropping useless data
df.loc[df.difference > 48, 'enrolled'] = 0
df = df.drop(columns = ["difference", "enrolled_date", "first_open"])
df.head()


# In[26]:


# Formatting the screen_list field

# Loading the dataset
top_screens = pd.read_csv("Downloads/datasets_87836_202131_top_screens.csv").top_screens.values

df["screen_list"] = df.screen_list.astype(str) + ','

for sc in top_screens:
    df[sc] = df.screen_list.str.contains(sc).astype(int)
    df["screen_list"] = df.screen_list.str.replace(sc+",", "")
    
print(top_screens)


# In[27]:


# Formatting the screen_list field
df["Other"] = df.screen_list.str.count(",")
df = df.drop(columns = ["screen_list"])


# In[28]:


# Funnels

# Saving
savings_screens = ['Saving1','Saving2', "Saving2Amount", 'Saving4', 'Saving5', 'Saving6', 'Saving7', 'Saving8', 'Saving9', 'Saving10']
df["'SavingCount"] = df[savings_screens].sum(axis = 1)
df = df.drop(columns = savings_screens)


# Credit
cm_screens = ['Credit1','Credit2', "Credit3Container", 'Credit3Dashboard']
df["CMCount"] = df[cm_screens].sum(axis = 1)
df = df.drop(columns = cm_screens)

# CC
cc_screens = ['CC1','CC1Category', "CC3"]
df["CCCount"] = df[cc_screens].sum(axis = 1)
df = df.drop(columns = cc_screens)

# Loan
loan_screens = ['Loan','Loan2', "Loan3", 'Loan4']
df["LoansCount"] = df[loan_screens].sum(axis = 1)
df = df.drop(columns = loan_screens)


# In[29]:


# Checking the result
df.head


# In[30]:


# Statistical info
df.describe()


# In[31]:


# Saving the new dataset
df.to_csv("new_appdata10.csv", index=False)


# ###### Data Preprocessing

# In[32]:


# Import packages
import time
from sklearn.model_selection import train_test_split


# In[33]:


# Loading dataset
original_new_appdata10 = pd.read_csv("new_appdata10.csv")
original_new_appdata10.head()


# In[34]:


# Creating a copy of the original_new_appdata10 dataset
dataset = original_new_appdata10.copy()


# In[35]:


# Creating the new dataset
response = dataset["enrolled"]
dataset = dataset.drop(columns = "enrolled")


# In[36]:


# Checking the result for the "response" dataframe
response.head()


# In[37]:


# Checking the result for the "dataset" dataframe
dataset.head()


# In[38]:


# Train, Test & Split the data
X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)

# Checking each shape of Test & Train set for X & y
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


# In[39]:


# Creating the User Train Identifier for the Train dataset
train_identifier = X_train["user"]

# Removing it from the dataset
X_train = X_train.drop(columns = "user")

# Creating the User Test Identifier for the Test dataset
test_identifier = X_test["user"]

# Removing it from the dataset
X_test = X_test.drop(columns = "user")


# In[40]:


# Feature Scaling

# Import package
from sklearn.preprocessing import StandardScaler

# Create the StandardScaler object
sc_X = StandardScaler()

# Fitting & Transforming X_train & X_test
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.fit_transform(X_test))


# In[41]:


# Setting columns & indexes

# Recuperating the columns from the original Training & Test set
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values

# Recuperating the indexes from the original Training & Test set
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values


# In[42]:


# Converting the Train & Test into the new Train & Test dataset
X_train = X_train2
X_test = X_test2


# ###### Model Building

# In[43]:


# Import packages
from sklearn.linear_model import LogisticRegression

# Create the LogisticRegression object
classifier = LogisticRegression(random_state = 0, penalty = "l1", solver='liblinear')


# In[44]:


# Fitting the Model
classifier.fit(X_train, y_train)


# In[45]:


# Prediction
y_pred = classifier.predict(X_test)


# In[46]:


# Import packages
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, log_loss


# In[47]:


# Accuracy
print("Accuracy: %0.4f" %  accuracy_score(y_test, y_pred))


# In[48]:


# Precision
print("Precision: %0.4f" %  precision_score(y_test, y_pred))


# In[49]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[50]:


# New dataframe
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))


# In[51]:


# Plotting Confusion Matrix
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[52]:


# Using k-fold cross validation for ensuring don't overfitting
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))


# In[53]:


# Analyzing Coefficients
pd.concat([pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["features"]),
pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])],axis = 1)


# #### Model Tuning and Regularization

# In[54]:


# Import package
from sklearn.model_selection import GridSearchCV


# In[63]:


# Select the Regularization Method
penalty = ["l1", "l2"]

# Create Regularization Hyperparameter space
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


# In[64]:


# Combine Parameters
parameters = dict(C=C, penalty=penalty)

# Create the GridSearch object
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=-1)
t0 = time.time()

# Fitting the Model
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Getting the results
best_res_accuracy = grid_search.best_score_
best_res_parameters = grid_search.best_params_
best_res_accuracy, best_res_parameters
grid_search.best_score_


# ##### End of the Model

# In[66]:


# Formatting Final Results
final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()
final_results["predicted_reach"] = y_pred
final_results = final_results[["user", "enrolled", "predicted_reach"]].reset_index(drop=True)


# In[67]:


# Looking at the result
final_results


# In[ ]:




