#!/usr/bin/env python
# coding: utf-8

# ### Loan Prediction Status Problem

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load data
df = pd.read_csv("Downloads/datasets_137197_325031_train_u6lujuX_CVtuZ9i.csv")
df.head()


# In[3]:


# Shape
df.shape


# In[4]:


# Statistical values 
df.describe()


# In[5]:


# All Statistical values
df.describe(include="all")


# In[6]:


# General info df
df.info()


# In[7]:


# Check null values in each feature
df.isnull().sum()


# In[8]:


# Change the type of 'Credit_History' to object becaues we can see that it is 1 or 0
df["Credit_History"] = df["Credit_History"].astype('O')
df.head()


# In[9]:


# Check by statistical values, all feature with type object
df.describe(include='O')


# In[10]:


# Drop 'Loan_ID' because it's not important for the model and it will just mislead the model
df.drop("Loan_ID", axis=1, inplace=True)


# In[11]:


# Chech the result
df.head()


# In[12]:


# Check duplicate values
df.duplicated().any()


# In[13]:


# look at the target percentage by plot
plt.figure(figsize=(8,6))
sns.countplot(df["Loan_Status"]);

print("The percentage of Y class : %.2f" % (df["Loan_Status"].value_counts()[0] / len(df)*100))
print("The percentage of X class : %.2f" % (df["Loan_Status"].value_counts()[1] / len(df)*100))


# ##### Look deeper in the data

# In[14]:


# Check features
df.columns


# In[15]:


# Go through the categorical features
df.head(1)


# In[16]:


# Credit_History

# Check count values by plot
grid = sns.FacetGrid(df,col="Loan_Status", size=3.2, aspect=1.6)
grid.map(sns.countplot, "Credit_History");

# we didn't give a loan for most people who got Credit History = 0
# but we did give a loan for most of people who got Credit History = 1
# so we can say if you got Credit History = 1 , you will have better chance to get a loan

# important feature


# In[17]:


# Gender

# Check count value by plot
grid = sns.FacetGrid(df, col="Loan_Status", size=3.2, aspect=1.6)
grid.map(sns.countplot, "Gender");


# most males got loan and most females got one too so (No pattern)


# In[18]:


# Married

# Ckeck count value by plot
grid = sns.FacetGrid(df, col="Loan_Status", size=3.2, aspect=1.6)
grid.map(sns.countplot, "Married")

# most people who get married did get a loan
# if you'r married then you have better chance to get a loan :)
# good feature


# In[19]:


# Married Alternative visualization

plt.figure(figsize=(15,5))
sns.countplot(x="Married", hue="Loan_Status", data=df)

# most people who get married did get a loan
# if you'r married then you have better chance to get a loan :)
# good feature


# In[20]:


# Dependents

plt.figure(figsize=(15,5))
sns.countplot(x="Dependents", hue="Loan_Status", data=df)

# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))
# good feature


# In[21]:


# Self_Employed

grid = sns.FacetGrid(df, col="Loan_Status", size=3.2, aspect=1.6)
grid.map(sns.countplot, "Self_Employed")


# In[22]:


# Education

grid = sns.FacetGrid(df, col="Loan_Status", size=3.2, aspect=1.6)
grid.map(sns.countplot, "Education")

# If you are graduated or not, you will get almost the same chance to get a loan (No pattern)
# Here you can see that most people did graduated, and most of them got a loan
# on the other hand, most of people who did't graduate also got a loan, but with less percentage from people who graduated

# not important feature


# In[23]:


# Property_Area

plt.figure(figsize=(15,5))
sns.countplot(x="Property_Area", hue="Loan_Status", data=df)

# We can say, Semiurban Property_Area got more than 50% chance to get a loan

# good feature


# In[24]:


# ApplicantIncome

plt.scatter(df["ApplicantIncome"], df["Loan_Status"])

# No pattern


# In[25]:


# The numerical data
df.groupby("Loan_Status").median() # Median because NOT AFFECTED with OUTLIERS

# we can see that when we got low median in CoapplicantInocme we got Loan_Status = N

# CoapplicantInocme is a good feature


# ##### Simple process for the data

# In[26]:


# Missing values
df.isnull().sum().sort_values(ascending=False)


# In[27]:


# We will separate the numerical columns from the categorical

cat_data = []
num_data = []

for i, c in enumerate (df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else:
        num_data.append(df.iloc[:, i])


# In[28]:


# Transpose 
cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()


# In[29]:


# Check the df cat_data
cat_data.head()


# In[30]:


# Check the num_data
num_data.head()


# In[31]:


# cat_data
# Fill every column with its own most frequent value you can use
cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
# No more missing data 
cat_data.isnull().sum().any()


# In[32]:


# num_data
# Fill every missing value with their previous value in the same column
num_data.fillna(method="bfill", inplace=True)
# No more missing data 
num_data.isnull().sum().any()


# #### Categorical columns

# In[33]:


# Import packages
from sklearn.preprocessing import LabelEncoder


# In[34]:


# Create a LabelEncoder statement object
le = LabelEncoder()
cat_data.head()


# In[35]:


# Transform the target column
target_values = {"Y": 0, "N": 1}
target = cat_data["Loan_Status"]
cat_data.drop("Loan_Status", axis=1, inplace=True)


# In[36]:


target = target.map(target_values)
target


# In[37]:


# Transform other columns
for i in cat_data:
    cat_data[i] = le.fit_transform(cat_data[i])


# In[38]:


# Check the result
target.head()


# In[39]:


# Check the result
cat_data. head()


# In[40]:


# Concatenate cat_data, num_data & target
df = pd.concat([cat_data, num_data, target], axis=1)
# Check the result
df.head()


# #### Train the data

# In[41]:


# Define y(target/dependent variable) & x(indipendent variables)
X = pd.concat([cat_data, num_data], axis=1)
y = target


# In[42]:


## Use StratifiedShuffleSplit to split the data, 
## taking into consideration that we will get the same ratio on the target column

# Import packages
from sklearn.model_selection import StratifiedShuffleSplit

# Create a StratifiedShuffleSplit statement
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# Create a loop throught training/testing/splitting data
for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

# Print the results
print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

# Almost same ratio
print('\nratio of target in y_train :',y_train.value_counts().values/ len(y_train))
print('ratio of target in y_test :',y_test.value_counts().values/ len(y_test))
print('ratio of target in original_data :',df['Loan_Status'].value_counts().values/ len(df))


# ##### Training the model

# In[43]:


# Use 4 different models for training

# Import packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[44]:


# Create a model object/statement
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}


# #### Building functions for evaluating the model
# 
# 1) **loss** : to evaluate the models by
# - precision
# - recall
# - f1
# - log_loss
# - accuracy_score
# 
# 
# 2) **train_eval_train** : to evaluate thr models in the *same data* that we train it on.
# 
# 3) **train_eval_cross** : to evaluate/compare the models using *different data* that we train the model on.
# 
# - StratifiedKFold

# In[45]:


# Loss

# Import packages
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score

# Define the function for evaluating differente scores
def loss(y_true, y_pred, retu=False):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
        
    else:
        print('pre: %.3f\n rec:%.3f\n f1: %.3f\n loss: %.3f\n acc: %.3f\n' % (pre, rec, f1, loss, acc))


# In[46]:


# Train_Eval_Train: same data that we train it on.

# Define the function for evaluating the same data
def train_eval_train(models, X, y):
    for name, model in models.items():
        print(name, ':')
        model.fit(X, y)
        loss(y, model.predict(X))
        print('-'*30)
        
train_eval_train(models, X_train, y_train)


# #### The best model is LogisticRegression at least for now, SVC is just memorizing the data so it is overfitting .

# In[47]:


# Shape of X trained
X_train.shape


# In[48]:


# Train_Eval_Cross

# Import packages
from sklearn.model_selection import StratifiedKFold

# Define the StratifiedKFold statement
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

# Define the cross function
def train_eval_cross(models, X, y, folds):
    # Change X & y to dataframe because we will use iloc (iloc don't work on numpy array)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    idx = ["pre", "rec", "f1", "loss", "acc"]
    for name, model in models.items():
        ls = []
        print(name, ":")
        
        for train, test in folds.split(X, y):
            model.fit(X.iloc[train], y.iloc[train])
            y_pred = model.predict(X.iloc[test])
            ls.append(loss(y.iloc[test], y_pred, retu=True))
        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0]) # [0] because we don't want to show the name of the column
        print('-'*30)
    
train_eval_cross(models, X_train, y_train, skf)
    


# #### SVC is just memorizing the data, and you can see that here DecisionTreeClassifier is better than LogisticRegression.

# In[49]:


# Some explanation of the above function

x = []
idx = ["pre", "rec", "f1", "loss", "acc"]

# Using LogisticRegression model
log = LogisticRegression()

# Define the loop
for train, test in skf.split(X_train, y_train):
    log.fit(X_train.iloc[train], y_train.iloc[train])
    ls = loss(y_train.iloc[test], log.predict(X_train.iloc[test]), retu=True)
    x.append(ls)
    
pd.DataFrame(x, columns=idx)

# Column 0 represents the 10 folds
# Row 0 represents the values (pre, rec, f1, loss, acc) for each fold


# In[50]:


# Mean of every column
pd.DataFrame(x, columns=idx).mean(axis=0)


# ### Improve the model
# 
# #### Features Engineer

# In[51]:


# Analyzing the features' correlation

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True)

# here we got 58% similarity between LoanAmount & ApplicantIncome 
# and that may be bad for our model so we will see what we can do


# In[52]:


# Note: the similarity between LoanAmount & ApplicantIncome is 58%
# that may be bad for the model

X_train.head()


# In[53]:


# Try to make diffrent operations on diffrent features

X_train["new_col"] = X_train["CoapplicantIncome"] / X_train["ApplicantIncome"]
X_train["new_col_2"] = X_train["LoanAmount"] * X_train["Loan_Amount_Term"]


# In[54]:


# Check againg the correlation between features
data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True)

# new_col 0.03 , new_col_2, 0.047
# not that much , but that will help us reduce the number of features


# In[55]:


# Drop useless features
X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)


# In[56]:


# Check the result
X_train.head()


# In[57]:


# Check the values for the Cross Evaluation
train_eval_cross(models, X_train, y_train, skf)


# ##### SVC is improving, but LogisticRegression is overfitting

# In[58]:


# Take a look at the value counts of every label
for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(), end='\n---------------------------------\n')


# ##### Working on the features that have varied values

# In[59]:


# new_col_2

# Note: we can see we got right_skewed
# Solution: take the logarithm of all the values, because when data is normally distributed that will help improving our model

# Import packages
from scipy.stats import norm

# Visualizing the skewed values
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

# Define the first plot to show the skewed's values before the logarithm
sns.distplot(X_train["new_col_2"], ax=ax[0], fit=norm)
ax[0].set_title("new_col_2 before log")

# Define the logarithm of all the values
X_train["new_col_2"] = np.log(X_train["new_col_2"])

# Create the second plot to show the skewed's values after the logarithm
sns.distplot(X_train["new_col_2"], ax=ax[1], fit=norm)
ax[1].set_title("new_col_2 after log")


# In[60]:


# Evaluate the models & do that continuously. So, don't need to mention that every time
train_eval_cross(models, X_train, y_train, skf)

# The models improved really good by just doing the previous step


# In[61]:


# new_col

# Most of the data are 0. Try to change other values to 1

print("before:")
print(X_train["new_col"].value_counts())

X_train["new_col"] = [x if x == 0 else 1 for x in X_train["new_col"]]
print('-'*50)

print('\nafter:')

print(X_train["new_col"].value_counts())


# In[62]:


# Evaluate/Check the models
train_eval_cross(models, X_train, y_train, skf)


# In[63]:


# Check the shapes, taking a look at the value counts of every label
for i in range (X_train.shape[1]):
    print(X_train.iloc[:, i].value_counts(), end='\n-------------------------------\n')


# #### Outliers

# In[64]:


# Plotting the outliers
sns.boxplot(X_train["new_col_2"])
plt.title("new_col_2 outliers", fontsize=15)
plt.xlabel('')


# In[65]:


# THRESHOLD
# this number is hyperparameter, as much as you reduce it, as much as you remove more points
                 # you can just try different values the deafult value is (1.5) it works good for most cases
                 # but be careful, you don't want to try a small number because you may loss some important information from the data .
                 # that's why I was surprised when 0.1 gived me the best result

# Define the threshold
threshold = 0.1
                        
# Create the new dataframe
new_col_2_out = X_train["new_col_2"]

# Quartile 25 &75
q25, q75 = np.percentile(new_col_2_out, 25), np.percentile(new_col_2_out, 75)
print("Quartile 25: {} , Quartile 75: {}".format(q25, q75))

iqr = q75 - q25
print("iqr: {} ".format(iqr))

cut = iqr * threshold
lower, upper = q25 - cut, q75 + cut
print("Cut off: {}". format(cut))
print("Lower: {}".format(lower))
print("Upper: {}".format(upper))

# Outliers
outliers = [x for x in new_col_2_out if x < lower or x > upper]
print("Nubers of Outliers: {}".format(len(outliers)))
print("Outliers: {}".format(outliers))

# Data before dropping outliers
data_outliers = pd.concat([X_train, y_train], axis=1)
print("\nLen before dropping outliers: ", len(data_outliers))

# Data after dropping the outliers
data_outliers = data_outliers.drop(data_outliers[(data_outliers["new_col_2"] > upper) | (data_outliers["new_col_2"] < lower)].index)

print("\nLen after dropping outliers: ", len(data_outliers))


# In[66]:


X_train = data_outliers.drop("Loan_Status", axis=1)
y_train = data_outliers["Loan_Status"]


# In[67]:


# Plotting the df without outliers
sns.boxplot(X_train["new_col_2"])
plt.title("new_col_2 without outliers", fontsize=15)
plt.xlabel('')


# In[68]:


# Evaluate/ Check again the model
train_eval_cross(models, X_train, y_train, skf)


# ##### Features selection

# In[69]:


# Check again the correlation values between features 
data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True)


# In[70]:


# Self_Employed got really bad corr (-0.00061) , let's try remove it and see what will happen
X_train.drop(["Self_Employed"], axis=1, inplace=True)


# In[71]:


# Evaluate/ Check again the model
train_eval_cross(models, X_train, y_train, skf)


# In[72]:


# Note: Considering the new results, after dropped the feature 'Self_Employed', it looks like Self_Employed was not important

# KNeighborsClassifier improved

# droping all the features except for Credit_History actually improved KNeighborsClassifier and didn't change anything in other models
# so you can try it by you self
# but don't forget to do that on testing data too


# In[73]:


# Check again the correlation values between features 
data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True)


# In[74]:


train_eval_cross(models, X_train, y_train, skf)


# ### Evaluate the models on Test_data
# Just repeat what we did in training data

# In[75]:


X_test.head()


# In[76]:


# Create a new dataframe, copying the original one
X_test_new = X_test.copy()


# In[77]:


# Try to make diffrent operations on diffrent features
x = []

X_test_new['new_col'] = X_test_new['CoapplicantIncome'] / X_test_new['ApplicantIncome']
X_test_new['new_col_2'] = X_test_new['LoanAmount'] * X_test_new['Loan_Amount_Term']

# Drop the original features which we insert into the new df above
X_test_new.drop(['CoapplicantIncome', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1, inplace=True)

X_test_new['new_col_2'] = np.log(X_test_new['new_col_2'])
X_test_new['new_col'] = [x if x==0 else 1 for x in X_test_new['new_col']]


# In[79]:


X_test_new.drop(["Self_Employed"], axis=1, inplace=True)


# In[80]:


# Test_data
X_test_new.head()


# In[81]:


# Train_data
X_train.head()


# In[82]:


# Evaluating the data
for name, model in models.items():
    print(name, end='\n')
    loss(y_test, model.predict(X_test_new))
    print('-'*40)


# #### Conclusion:
# 
# what ever we do, our recall score will not improving , maybe because we don't have a good amount of data, so I think if we got more data and we try more complex models our accuracy will improve,I am not sure about this, so please if I made any mistakes in this kernel , or if you have any suggestions which can improve the accuracy please feel free to share it with us in the comments .
