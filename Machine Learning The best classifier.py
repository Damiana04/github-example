#!/usr/bin/env python
# coding: utf-8

# ##### Data Exploratory

# In[1]:


# Import packages
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Load data
df = pd.read_csv("Downloads/loan_train.csv")
df.head()


# In[5]:


# Shape
df.shape


# In[6]:


# Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[24]:


# Column names
df.columns


# ##### Data visualization and pre-processing

# In[10]:


# Counting values in the dataset
df["loan_status"].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection

# ##### Exploring dataset by Visualizzation

# In[41]:


# Counting by Gender & Loan_status
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans, while only 73 % of males pay there loan

# In[17]:


# Import library
import seaborn as sns

# Plotting by Principal, Gender & Loan_status
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set2", col_wrap=2) # col_wrap divides the colum in 2 
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[19]:


# Plotting by Principal, Gender & Loan_status
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set2", col_wrap=2) # col_wrap divides the colum in 2 
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[21]:


# Counting values in the dataset
df["Gender"].value_counts()


# #### Pre-processing: Feature selection/extraction
# 
# ###### The day of the week people get the loan

# In[30]:


df[["dayofweek", "effective_date", "Gender"]].head(20)


# In[35]:


# Starting date in the dataset (first date in the range time)
df["effective_date"].min()


# In[36]:


# Ending date in the dataset (last date in the range time)
df["effective_date"].max()


# In[37]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set2", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# Note: People who get the loan at the end of the week dont pay it off.

# In[43]:


# Creating a new column "weekend" with binarized value 1 for range dayofweek from 4 to 7 and 0 from 1 to 3
# Feature binarization to set a threshold values less then day 4
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ##### Convert Categorical features to numerical values

# In[44]:


# Counting by Gender & Loan_status
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[45]:


# Converting the categorical features male to 0 and female to 1
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ##### One Hot Encoding

# In[46]:


# Counting by Education & Loan_status
df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# ##### Feature befor One Hot Encoding

# In[48]:


# Exploring data before One Hot Encoding
df[['Principal','terms','age','Gender','education']].head()


# ##### Use one hot encoding technique to convert categorical varables to binary variables and append them to the feature Data Frame

# In[50]:


# Creating a df by One Hot Encoding highliting the Education roots
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ##### Feature selection: definding X & y for the next steps 

# In[51]:


# Defind feature sets in X
X = Feature
X[0:5]


# In[52]:


# Defind labels sets in y
y = df['loan_status'].values
y[0:5]


# #### Normalize Data
# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[1]:


# Normalize/Fit/Transform the dataset
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### CLASSIFICATION
# 
# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model You should use the following algorithm:
# 
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# __ Notice:__
# 
# You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# You should include the code of the algorithm in the following cells.

# ### K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.
# 
# 
# warning: You should not use the loan_test.csv for finding the best k, however, you can split your train_loan.csv into train and test to find the best k.

# In[56]:


# Importing packages
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[57]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
X_train


# In[58]:


from sklearn.neighbors import KNeighborsClassifier

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc #print accuracy array for eack k


# In[85]:


plt.plot(range(1,Ks),mean_acc,'r')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.70)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# ### Decision Tree

# In[60]:


from sklearn.tree import DecisionTreeClassifier
df


# In[61]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)


# In[62]:


#INSTALLATIONS TO VIEW THE DECISION TREE
get_ipython().system('conda install -c conda-forge pydotplus -y')


# In[63]:


get_ipython().system('conda install -c conda-forge python-graphviz -y')


# In[64]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[3: 11]
targetNames = df["loan_status"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# ### Support Vector Machine

# In[65]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)


# In[66]:


yhatsvm = clf.predict(X_test)
yhatsvm [0:5]


# In[68]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat)
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=df["loan_status"].unique().tolist(),normalize= False,  title='Confusion matrix')


# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[70]:


yhatl = LR.predict(X_test)
yhatl


# In[71]:


yhat_prob = LR.predict_proba(X_test)


# ### Model Evaluation using Test set

# In[72]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[80]:


# Load data
test_df = pd.read_csv("Downloads/loan_train.csv")
test_df.head()


# In[81]:


#test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
print(test_df.dtypes)
import numpy as np
test_df['Gender'].replace(to_replace=['male','female'], value=[np.int64(0),np.int64(1)],inplace=True)

test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)
Feature = test_df[['Principal','terms','age','Gender']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()
X2 = Feature
y2 = test_df['loan_status'].values
X2= preprocessing.StandardScaler().fit(X2).transform(X2)


# ### Report

# In[86]:


print( "KNN: The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[82]:


predTree = drugTree.predict(X_test)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[83]:


# for SVM
classification_report(y_test,yhatsvm )
print('SVM ACCY:', classification_report(y_test, yhatsvm))


# In[84]:


#for logit
classification_report(y_test, yhatl)
print('Logit', classification_report(y_test, yhatl))


# In[ ]:




