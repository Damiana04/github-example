#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection with Deep Neural Network

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


# Random seeds
np.random.seed(2)


# In[3]:


# Load the dataset
df_original = pd.read_csv("Downloads/creditcard.csv")
df_original.head()


# In[4]:


# Creating a copy of the original dataset
df = df_original.copy()
df.head()


# ##### Data exploration

# In[5]:


# General info
df.info()


# In[6]:


# Statistical values including missing values
df.describe(include="all")


# In[7]:


# Checking NaN values
df.isnull().sum()


# In[8]:


# Unique values into the "Class" feature
df["Class"].unique()


# In[9]:


# Counting values into the "Class" feature
df.Class.value_counts


# In[10]:


# Unique values into the "Amount" feature
df["Amount"].unique()


# In[11]:


# Counting values into the "Amount" feature
df.Amount.value_counts


# #### Pre-processing

# In[12]:


# Import library
from sklearn.preprocessing import StandardScaler

# Normalizing the "Amount" feature
df["normalizeAmount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
df = df.drop(["Amount"], axis=1)

# Checking the result
df.head()


# In[13]:


# Removing the "Time" feature
df = df.drop(["Time"], axis=1)
df.head()


# In[14]:


# Definding X indipendent variables keeping all columns but skiping the coulumns "Class"
X = df.iloc[:, df.columns != "Class"]

# Definding y target variable as "Class" feature
y = df.iloc[:, df.columns == "Class"]


# ##### Train_Test_Split

# In[15]:


# Import package
from sklearn.model_selection import train_test_split

# Train_Test_Split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.30, random_state=0)

# Printing the shapes
print("X_train shape is: ", X_train.shape)
print("X_test shape is: ", X_test.shape)
print("y_train shape is: ", y_train.shape)
print("y_test shape is: ", y_test.shape)


# In[16]:


# Storing in array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ##### Deep Neural Network

# In[17]:


# Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[18]:


# Definding the Model 
model = Sequential([
    Dense(units=16, input_dim=29, activation="relu"),
    Dense(units=24, activation="relu"),
    Dropout(0.5),
    Dense(units=20, activation="relu"),
    Dense(units=24, activation="relu"),
    Dense(1, activation="sigmoid")
])


# In[19]:


# Checking the result 
model.summary()


# ##### Training 

# In[20]:


# Metrics
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

# Fitting
model.fit(X_train, y_train, batch_size=15, epochs=5)


# In[21]:


# Evaluating the test dataset
score = model.evaluate(X_test, y_test)


# In[22]:


# Printing the accuracy's result
print(score)


# ##### Confusion Matrix

# In[35]:


# Import packages
import itertools
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Definding the function for plotting Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[36]:


# Prediction
y_pred = model.predict(X_test)

# Transforming y_test dataset in pandas
y_test = pd.DataFrame(y_test)


# In[37]:


# Definding Confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred.round())
print(cnf_matrix)


# In[38]:


# Plotting Confusion Matrix by calling the function
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[39]:


# Prediction & Confusion matrix with X & y parameters
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# #### Random Forest

# In[40]:


# Import package
from sklearn.ensemble import RandomForestClassifier


# In[41]:


# Definding the Random Forest
random_forest = RandomForestClassifier(n_estimators=100)


# In[46]:


# Transforming y_train dataset in pandas for fitting the model
y_train = pd.DataFrame(y_train) 


# In[47]:


# Fitting the Random Forest model
random_forest.fit(X_train, y_train.values.ravel())


# In[49]:


# Prediction Random Forest 
y_pred = random_forest.predict(X_test)


# In[50]:


# Accuracy
random_forest.score(X_test, y_test)


# ##### Confusion matrix

# In[51]:


# Import packages
import itertools
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Definding the function for plotting Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[52]:


# Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[53]:


# Prediction X
y_pred = random_forest.predict(X)

# Confusion Matrix X
cnf_matrix = confusion_matrix(y, y_pred.round())
print(cnf_matrix)

# Plotting Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# ##### Decision Tree

# In[54]:


# Import package
from sklearn.tree import DecisionTreeClassifier


# In[55]:


# Definding the Decision Tree model
decision_tree = DecisionTreeClassifier()


# In[56]:


# Fitting the Decision Tree model
decision_tree.fit(X_train, y_train.values.ravel())


# In[57]:


# Prediction in Decision Tree 
y_pred = decision_tree.predict(X_test)


# In[58]:


# Accuracy 
decision_tree.score(X_test, y_test)


# In[59]:


# Definding the function for plotting Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[60]:


# Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# Plotting Confusion Matrix 
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[61]:


# Prediction Decision Tree with X & y parameters
y_pred = decision_tree.predict(X)
y_expected = pd.DataFrame(y)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
print(cnf_matrix)

# Plotting Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# ##### Undersampling

# In[62]:


# Definding the Fraud records
fraud_indices = np.array(df[df.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)


# In[63]:


# Definding the Normal records
normal_indices = np.array(df[df.Class == 0].index)


# In[64]:


# Creating a Random Sample
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)

# Converting into a Np Array
random_normal_indices = np.array(random_normal_indices)

# Cheching the number of Normal transactions
print(len(random_normal_indices))


# In[66]:


# Creating the Undersampling array
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# Cheching the len
print(len(under_sample_indices))


# In[67]:


# Creating the new dataset
under_sample_data = df.iloc[under_sample_indices, :]


# In[69]:


# Definding X & y 
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != "Class"]
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == "Class"]


# In[71]:


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3)

# Printing the shapes
print("X_train shape is: ", X_train.shape)
print("X_test shape is: ", X_test.shape)
print("y_train shape is: ", y_train.shape)
print("y_test shape is: ", y_test.shape)


# In[72]:


# Transforming from pandas to np.array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[73]:


# Checking the result
model.summary()


# In[74]:


# TRAINING

# Metrics
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

# Fitting
model.fit(X_train, y_train, batch_size=15, epochs=5)


# In[78]:


# Prediction with X_test & y_test parameters
y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
print(cnf_matrix)

# Plotting Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[79]:


# Prediction into the intere dataset 
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
print(cnf_matrix)

# Plotting Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# ##### Smote

# In[81]:


## bash
get_ipython().system('pip install -U imbalanced-learn')


# In[82]:


# Import library
from imblearn.over_sampling import SMOTE


# In[83]:


# Fitting the SMOTE
X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())


# In[84]:


# Transforming the parameters from np.array to pandas
X_resample = pd.DataFrame(X_resample)
X_resample = pd.DataFrame(X_resample)


# In[85]:


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.3)

# Printing the shapes
print("X_train shape is: ", X_train.shape)
print("X_test shape is: ", X_test.shape)
print("y_train shape is: ", y_train.shape)
print("y_test shape is: ", y_test.shape)


# In[86]:


# Transforming from pandas to np.array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[87]:


# Checking the result
model.summary()


# In[88]:


# TRAINING

# Metrics
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

# Fitting
model.fit(X_train, y_train, batch_size=15, epochs=5)


# In[89]:


# Prediction with X_test & y_test parameters
y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
print(cnf_matrix)

# Plotting Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[90]:


# Prediction with X_test & y_test parameters
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
print(cnf_matrix)

# Plotting Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[ ]:




