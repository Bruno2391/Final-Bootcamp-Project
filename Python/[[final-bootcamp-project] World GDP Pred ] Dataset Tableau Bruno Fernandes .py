#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


gdp_pred = pd.read_excel(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets Clean\gdp_pred.xlsx")
gdp_pred


# In[3]:


gdp_capita_pivot = pd.read_excel(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets Clean\gdp_capita_pivot.xlsx")
gdp_capita_pivot


# In[4]:


gdp_ppp_pivot = pd.read_excel(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets Clean\gdp_ppp_pivot.xlsx")
gdp_ppp_pivot


# In[5]:


gdp_growth_pivot = pd.read_excel(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets Clean\gdp_growth_pivot.xlsx")
gdp_growth_pivot


# In[6]:


capita_growth = pd.merge(gdp_capita_pivot, gdp_growth_pivot, on=['country', 'year'])
capita_growth


# In[7]:


capita_growth.to_excel('capita_growth_pivot.xlsx', index=False)

