#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")
diamonds = sns.load_dataset("diamonds")
diamonds.head()



# In[2]:


diamonds.shape


# In[3]:


diamonds.describe()


# In[4]:


diamonds.describe(exclude=np.number)


# In[5]:


import sklearn;
from sklearn.model_selection import train_test_split

# Extract feature and target arrays
X, y = diamonds.drop('price', axis=1), diamonds[['price']]

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')

X.dtypes


# In[6]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

import xgboost as xgb

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)

from sklearn.metrics import mean_squared_error

preds = model.predict(dtest_reg)

rmse = mean_squared_error(y_test, preds)

print(f"RMSE of the base model: {rmse:.3f}")


# In[7]:


n = 10000
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=50,
   early_stopping_rounds=50
)


# In[8]:


# cross validation with folding

params = {"objective": "reg:squarederror", "tree_method": "hist"}
n = 1000

results = xgb.cv(
   params, dtrain_reg,
   num_boost_round=n,
   nfold=5,
   early_stopping_rounds=20
)


# In[9]:


results.head()


# In[10]:


best_rmse = results['test-rmse-mean'].min()
best_rmse


# In[11]:


# XGBoost Classification
get_ipython().run_line_magic('%time', '')

from sklearn.preprocessing import OrdinalEncoder

X, y = diamonds.drop("cut", axis=1), diamonds[['cut']]

# Encode y to numeric
y_encoded = OrdinalEncoder().fit_transform(y)

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to pd.Categorical
for col in cats:
   X[col] = X[col].astype('category')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1, stratify=y_encoded)

# Create classification matrices
dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "multi:softprob", "tree_method": "hist", "num_class": 5}
n = 1000

results = xgb.cv(
   params, dtrain_clf,
   num_boost_round=n,
   nfold=5,
   metrics=["mlogloss", "auc", "merror"],
)
results.keys()


# In[12]:


results['test-auc-mean'].max()


# In[ ]:





# In[ ]:




