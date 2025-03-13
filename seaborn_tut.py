#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

warnings.filterwarnings("ignore")
#diamonds = sns.load_dataset("diamonds")
    #pd.read_csv('')
crash_df = sns.load_dataset('car_crashes')
crash_df.head()


# In[7]:


sns.distplot(crash_df['not_distracted'],kde=False)


# In[11]:


sns.jointplot(x='speeding', y='alcohol', data=crash_df, kind='reg')


# In[12]:


sns.pairplot(crash_df)


# In[15]:


data = crash_df.drop(columns=['abbrev'])
confusion_matrix = data.corr()
sns.heatmap(confusion_matrix, annot=True, cmap='coolwarm')


# In[ ]:




