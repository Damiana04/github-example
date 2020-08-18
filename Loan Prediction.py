#!/usr/bin/env python
# coding: utf-8

# #### Loan Prediction

# In[1]:


# Import packages
import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore')


# In[2]:


# Load TRAIN dataset
train = pd.read_csv("datasets_14289_19202_train_u6lujuX_CVtuZ9i.csv")
train.head()


# In[3]:


# Shape TRAIN dataset
train.shape


# In[4]:


# Load TEST dataset
test = pd.read_csv('datasets_137197_325031_test_Y3wMUE5_7gLdaTN.csv')
test.head()


# In[5]:


# Shape Test dataset
test.shape


# In[6]:


# Create a copy for both dataframe
train_original = train.copy()
test_original = test.copy()


# In[7]:


# Columns Train dataset
train.columns


# In[8]:


# Columns Test dataset
test.columns


# In[9]:


# General info Train dataset
train.info()


# In[10]:


# General info Test dataset
test.info()


# In[11]:


# Statistical values Train dataset, including all
train.describe()


# In[12]:


# Statistical values Test dataset, including all
test.describe()


# In[13]:


# Shapes Train & Test
print(train.shape)
print(test.shape)


# ##### Loan_Status as Target Variable 

# In[14]:


# Exploring 'Loan_Status'
train['Loan_Status'].value_counts()


# In[15]:


# 'Loan_Status' visualization
train['Loan_Status'].value_counts().plot.bar()


# ##### Independent Variable (Categorical)

# In[16]:


# Visualizing all Categorical Variables

# Gender feature
plt.figure(1)
plt.subplot(221)
train["Gender"].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

# Married feature
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

# Self_Employed feature
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

# Credit_History feature
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

# Visualize the results
plt.show()


# ##### It can be inferred from the above bar plots that:
# 
# - 80% applicants in the dataset are male.
# - Around 65% of the applicants in the dataset are married.
# - Around 15% applicants in the dataset are self employed.
# - Around 85% applicants have repaid their debts.

# ##### Independent Variable (Ordinal)

# In[17]:


# Visualize all Independent Variable (Ordinal)
plt.figure(1)

# 'Dependents' feature
plt.subplot(131)
train["Dependents"].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

# 'Education' feature
plt.subplot(132)
train["Education"].value_counts(normalize=True).plot.bar(title= 'Education')

# 'Property_Area' feature
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

# Visualize the results
plt.show()


# ##### Following inferences can be made from the above bar plots:
# 
# - Most of the applicants don’t have any dependents.
# - Around 80% of the applicants are Graduate.
# - Most of the applicants are from Semiurban area.

# ##### Independent Variable (Numerical)

# In[18]:


# 'ApplicantIncome' Distribution Visualization
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);

# 'ApplicantIncome' Outliers Visualization
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))

# Visualize the results
plt.show()


# In[19]:


# Analyzing the 'ApplicantIncome' Outliers divided by 'Education'
train.boxplot(column='ApplicantIncome', by= 'Education', figsize=(10,8))
plt.suptitle("") 


# In[20]:


# Coapplicant Distribution Visualization
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

# 'Coapplicant' Outliers Visualization
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))

# Visualize the results
plt.show()


# In[21]:


# 'LoanAmount' Distribution Visualization
plt.figure(1)
plt.subplot(121)
sns.distplot(train['LoanAmount']);

# 'LoanAmount' Outliers Visualization
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

# Visualize the results
plt.show()


# ##### Categorical Independent Variable vs Target Variable

# In[22]:


# 

# Gender feature
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Married feature
Married = pd.crosstab(train['Married'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Dependents feature
Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Education feature
Education = pd.crosstab(train['Education'], train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Self_Employed feature
Self_Employed = pd.crosstab(train['Self_Employed'], train['Self_Employed'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Credit_History feature
Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Property_Area feature
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# Visualize the results
plt.show()


# ##### Numerical Independent Variable vs Target Variable

# In[23]:


# ApplicantIncome vs Loan Status
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[24]:


# Making bins for the 'ApplicantIncome' variable based on the values
bins = [0,2500,4000,6000,81000]
groups = ['Low','Average','High', 'Very high']
train['Income_bin'] = pd.cut(train['ApplicantIncome'], bins, labels=groups)

# Visualization
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')


# In[25]:


# Making bins for the 'CoapplicantIncome' variable based on the values
bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

# Visualization
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# In[26]:


# Effect on the Loan Status, combining Applicant Income and Coapplicant Income
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high']

train['Total_Income_bin'] = pd.cut(train['Total_Income'], bins, labels=groups)
Total_Income_bin = pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# In[27]:


# Loan amount variable
bins=[0,100,200,700] 
group=['Low','Average','High'] 

train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')


# In[28]:


# Drop the bins which we created for the exploration part
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)


# In[29]:


# Replace 3+ in dependents variable to 3 to make it a numerical variable in both train & test dataset
train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True)


# In[30]:


# Replace the target variable’s categories into 0 and 1 for finding the correlation
train['Loan_Status'].replace('N', 0,inplace=True)


# In[31]:


# Replace N with 0 and Y with 1 because logistic regression takes only numeric values
train['Loan_Status'].replace('Y', 1,inplace=True)


# In[32]:


# Correlation Matrix
matrix = train.corr() 
f, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix, vmax=.8, square=True, cmap='BuPu')


# The most correlated variables are 'ApplicantIncome - LoanAmount' and 'Credit_History - Loan_Status'. 
# LoanAmount is also correlated with CoapplicantIncome.

# ##### Missing values 

# In[33]:


# Counting Missing values for each feature in the Train dataset
train.isnull().sum().sort_values(ascending=False)


# In[34]:


# Filling Missing Values in the Train dataset
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[35]:


# Checking again the Missing values in the Train dataset
train.isnull().sum().any()


# In[36]:


# Other Missing values
train.isnull().sum()


# In[37]:


# Look at the value count of the Loan amount term variable
train['Loan_Amount_Term'].value_counts()


# The value of 360 is repeating the most. So we will replace the missing values in this variable using the mode of this variable

# In[38]:


# Replace missing value in Loan_Amount_Term with the mode(value of 360)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[39]:


# LoanAmount variable as a numerical variable,use mean or median to impute the missing values
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[40]:


# Checking again the Missing values in the Train dataset
train.isnull().sum()


# In[41]:


# Checking Missing Values in the Test dataset
test.isnull().sum()


# In[42]:


# Filling Missing Values in the Test dataset
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)


# In[43]:


# Checking again the Missing values in the Test dataset
test.isnull().sum()


# ##### Outliers

# In[44]:


# Mean in the LoanAmount - Train dataset with outliers
train['LoanAmount'].mean()


# In[45]:


# Standard Deviation in the LoanAmount - Train dataset with outliers
train['LoanAmount'].std()


# In[46]:


# Mode in the LoanAmount - Train dataset with outliers
train['LoanAmount'].mode()


# In[47]:


# Median in the LoanAmount - Train dataset with outliers
train['LoanAmount'].median()


# In[48]:


# Distribution in the LoanAmount - Train dataset with outliers
train['LoanAmount'].hist(bins=20)


# In[49]:


# Mean in the LoanAmount - Test dataset with outliers
test['LoanAmount'].mean()


# In[50]:


# Standard Deviation in the LoanAmount - Test dataset with outliers
test['LoanAmount'].std()


# In[51]:


# Mode in the LoanAmount - Test dataset with outliers
test['LoanAmount'].mode()


# In[52]:


# Median in the LoanAmount - Test dataset with outliers
test['LoanAmount'].median()


# In[53]:


# Distribution in the LoanAmount - Test dataset with outliers
test['LoanAmount'].hist(bins=20)


# ##### Outlier Treatment

# In[54]:


# Removing Outliers in the LoanAmount - Train dataset
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)


# In[55]:


# LoanAmount - Train dataset after removing outliers
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount_log'])
plt.subplot(122)
train['LoanAmount_log'].plot.box(figsize=(16,5))


# In[56]:


# Removing Outliers in the LoanAmount - Test dataset
test['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'].hist(bins=20)


# In[57]:


# LoanAmount - Test dataset after removing outliers
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(test['LoanAmount_log'])
plt.subplot(122)
test['LoanAmount_log'].plot.box(figsize=(16,5))


# ##### Evaluation Metrics for Classification Problems

# ##### Preprocessing before starting prediction

# In[58]:


# Drop useless feature as Loan_ID
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)


# In[59]:


# Define x(indipendent variables/features) & y(dependent variable/target)
X = train.drop('Loan_Status', 1)
y = train.Loan_Status


# In[60]:


# Dummies for categorical variables
X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# ###### Train_Test_Split

# In[61]:


# Import package
from sklearn.model_selection import train_test_split

# Process Train_Test_Split 70% in traing, 30% testing
X_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)


# ##### Applying Logistic Regression Model

# In[62]:


# Import package
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

# Define the LogisticRegression statement/object
model = LogisticRegression()

# Fit/Train the model with the train dataset
model.fit(X_train, y_train)


# ##### Prediction phase

# In[63]:


# Predicting the model with C (regulation strenght) for reducing overfitting in the TRAIN dataset
pred_cv = model.predict(x_cv)


# C parameter represents inverse of regularization strength. Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. Smaller values of C specify stronger regularization.

# In[64]:


# Calculating the Accuracy
accuracy_score(y_cv, pred_cv)


# In[65]:


# Predicting the model with C (regulation strenght) for reducing overfitting in the TEST dataset
pred_test = model.predict(test)


# ##### Logistic Regression using stratified k-folds cross validation

# In[66]:


# Import package
from sklearn.model_selection import StratifiedKFold


# In[67]:


# Define 5 folds
i = 1 

# Define the StratifiedKFold statement/object
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

# Loop throught to automate the cross validation process
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold{}'.format (i, kf.n_splits))
    xtr,xvl = X.loc[train_index], X.loc[test_index]
    ytr,yvl = y.loc[train_index], y.loc[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr,ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl,pred_test)
    print("accuracy_score", score)
    i+=1 
    pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:,1]


# In[68]:


# The Roc Curve

# Import package
from sklearn import metrics

# Define the statement
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label="validation, auc="+str(auc))
plt.xlabel("False Positive Rate", fontsize=(15))
plt.ylabel("True Positive Rate", fontsize=(15))
plt.legend(loc=4)
plt.show()


# ##### Features that can affect the target

# In[69]:


# Total_Income
train["Total_Income"] = train["ApplicantIncome"] + train["CoapplicantIncome"]
test["Total_Income"] = test["ApplicantIncome"] + test["CoapplicantIncome"]


# In[70]:


# Visualization of Total Income's distribution in the TRAIN dataset
sns.distplot(train["Total_Income"])


# In[71]:


# Improving the right skewed of Total_Income in the TRAIN dataset. Using the log transformation to make the distribution normal
train["Total_Income_log"] = np.log(train["Total_Income"])

# Check the trasformation's result
sns.distplot(train["Total_Income_log"])


# In[72]:


# Visualization of Total Income's distribution in the TEST dataset
sns.distplot(test["Total_Income"])


# In[73]:


# Improving the right skewed of Total_Income in the TEST dataset. Using the log transformation to make the distribution normal
test["Total_Income_log"] = np.log(test["Total_Income"])

# Check the trasformation's result
sns.distplot(test["Total_Income_log"])


# In[74]:


# EMI: monthly amount to be paid by the applicant to repay the loan
train["EMI"] = train["LoanAmount"] / train["Loan_Amount_Term"]
test["EMI"] = test["LoanAmount"] / test["Loan_Amount_Term"]


# In[75]:


# Visualization of EMI's distribution in the TRAIN dataset
sns.distplot(train["EMI"])


# In[76]:


# Balance Income: 
train["Balance Income"] = train["Total_Income"] - train["EMI"] * 1000
test["Balance Income"] = test["Total_Income"] - test["EMI"] * 1000


# In[77]:


# Visualization of Balance Income's distribution in the TRAIN dataset
sns.distplot(train["Balance Income"])


# In[78]:


# Visualization of Balance Income's distribution in the TEST dataset
sns.distplot(test["Balance Income"])


# ###### Drop the variables which we used to create these new features. 
# 
# Reasons:
# - the correlation between those old features and these new features will be very high 
# - the logistic regression could assume that the variables are not highly correlated
# - it will help in reducing the noise from the dataset

# In[79]:


# Drop the variables which we used to create these new features
train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# In[80]:


xtr.isnull().sum()


# ##### Preparing data for the models: 
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - XGBoost

# In[81]:


xtr.isnull().sum()


# In[82]:


ytr.isnull().sum()


# In[83]:


xtr.info()


# In[84]:


xtr.columns


# In[85]:


xtr["LoanAmount_log"].astype('int')
xtr["Credit_History"].astype('int')


# In[86]:


print(np.any(np.isnan(xtr)))
print(np.all(np.isfinite(xtr)))


# In[87]:


np.isfinite(xtr).sum()


# In[88]:


np.isnan(xtr).sum() 


# In[91]:


# Define X & y
X = train.drop("Loan_Status", 1)
y = train.Loan_Status


# In[92]:


# Logistic Regression
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]
    
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print("accuracy_score", score)
    
    i+= 1
    
    pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:,1]


# ##### Decision Tree

# In[93]:


# Import package 
from sklearn import tree


# In[95]:


# Define the Decision Tree statement
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y.loc[train_index], y.loc[test_index]
    
    model = tree.DecisionTreeClassifier(random_state=1)
    model.fit(xtr,ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score',score)
    
    i+=1
    pred_test = model.predict(test)  


# ##### Random Forest

# In[97]:


# Import package
from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest Classifier statement
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y.loc[train_index], y.loc[test_index]
    
    model = RandomForestClassifier(random_state=1, max_depth=10)
    model.fit(xtr,ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score',score)
    
    i+=1
    pred_test = model.predict(test)


# ##### Grid Search

# In[109]:


# Import package
from sklearn.model_selection import GridSearchCV

# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

# Define the Grid Search statement
grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)

# Train, Test & Split
from sklearn.model_selection import train_test_split 
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=3, random_state=1)

# Fit the grid search model 
grid_search.fit(x_train,y_train)

GridSearchCV(cv=None, error_score='raise'),
estimator = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_node=None, min_impurity_decrease=0.0, min_impurity_split=None, min_sample_leaf=1, min_sample_splits=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=1, verbose=0, warm_start=False), 
fit_params=None, 
iid=False, 
n_jobs=1,
param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},
pre_dispatch = '2*n_jobs', 
refit=True, 
return_train_score='warn', 
scoring=None, 
verbose=0

# Estimating the optimized value 
grid_search.best_estimator_
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,            
min_samples_leaf=1, min_samples_split=2,            
min_weight_fraction_leaf=0.0, n_estimators=41, n_jobs=1,            
oob_score=False, random_state=1, verbose=0, warm_start=False)


# In[ ]:




