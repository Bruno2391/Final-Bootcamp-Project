#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import geopandas as gpd
import math


# In[2]:


# Importing the dataset 
gdp_capita = pd.read_csv(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets\gdp_per_capita.csv")
gdp_capita


# In[3]:


# Rename of the first column to country 
gdp_capita.rename(columns={'GDP, current prices (Billions of U.S. dollars)': 'Country'}, inplace=True)
gdp_capita


# In[4]:


# Rename columns to lower case and replace spaces
cols = []
for i in range(len(gdp_capita.columns)):
    cols.append(gdp_capita.columns[i].lower().replace(' ', ''))
gdp_capita.columns = cols
gdp_capita


# In[5]:


#checking the struture of the dataset
gdp_capita.shape


# In[6]:


# Dropping from the row 196 until 230
gdp_capita.drop(range(196, 230), inplace=True)
gdp_capita


# In[7]:


# Checking a summary of the dataset to understand the structure and content of the dataset, as null values, and data types
gdp_capita.info()


# In[8]:


# checking all the rows of the dataset
pd.set_option('display.max_rows', None)
print(gdp_capita)


# In[9]:


# checking the null values of each feature
gdp_capita.isna().sum()


# In[10]:


# Checking duplicates 
gdp_capita.duplicated().sum()


# In[11]:


# Checking statistics of the dataset
gdp_capita.describe().T


# In[12]:


# Checking the columns of the dataset 
gdp_capita.columns


# In[13]:


# Checking unique values of each feature
gdp_capita.nunique()


# In[14]:


# Printing the dataset
gdp_capita


# In[15]:


# Plotting the trend of GDP of each country 
for index, row in gdp_capita.iterrows():
    country_name = row['country']
    gdp_data = row.values[1:]

    plt.figure(figsize=(8, 6))
    plt.plot(gdp_capita.columns[1:], gdp_data)
    plt.title(f'GDP per capita {country_name}')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.xticks(rotation='vertical')
    plt.show()


# In[16]:


# ploting the GDP trend of all the countries
plt.figure(figsize=(10, 6))
for index, row in gdp_capita.iterrows():
    country_name = row['country']
    gdp_data = row.values[1:]

    plt.plot(gdp_capita.columns[1:], gdp_data, label=country_name)

plt.title('GDP per Capita for All Countries')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout() 
plt.show()


# In[17]:


# Printing the dataset 
gdp_capita


# In[18]:


# Extracting the dataset clean 
gdp_capita.to_excel('gdp_capita_clean.xlsx', index=False)


# In[19]:


# Creating a new column with the sum of the avg of the growth and the last year 
avg_growth = gdp_capita.loc[:, '1981':'2022'].mean(axis=1)
sum_avg_growth_last_year = avg_growth + gdp_capita['2023']
gdp_capita['sum_avg_growth_last_year'] = sum_avg_growth_last_year
gdp_capita


# In[20]:


# Defining the X and y axis 
columns_to_drop = ['country','sum_avg_growth_last_year']
X = gdp_capita.drop(columns=columns_to_drop)
y = gdp_capita['sum_avg_growth_last_year']


# In[21]:


# Spliting the X and y 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Testing the LR model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)


# In[23]:


# Printing predictions
predictions_lr


# In[24]:


# Testing the RF model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)


# In[25]:


# Printing predictions
predictions_rf


# In[26]:


# Testing the GB model
from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)


# In[27]:


# Printing predictions
predictions_gb


# In[28]:


# Evaluating the LR model
mae_lr = mean_absolute_error(y_test, predictions_lr)
mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)


# In[29]:


# Evaluating the RF model
mae_rf = mean_absolute_error(y_test, predictions_rf)
mse_rf = mean_squared_error(y_test, predictions_rf)
r2_rf = r2_score(y_test, predictions_rf)


# In[30]:


# Evaluating the GB model
mae_gb = mean_absolute_error(y_test, predictions_gb)
mse_gb = mean_squared_error(y_test, predictions_gb)
r2_gb = r2_score(y_test, predictions_gb)


# In[31]:


# Printing the evaluations of the models
rmse_lr = np.sqrt(mean_squared_error(y_test, predictions_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, predictions_rf))
rmse_gb = np.sqrt(mean_squared_error(y_test, predictions_gb))

print("Linear Regression Metrics:")
print("Mean Absolute Error:", mae_lr)
print("Mean Squared Error:", mse_lr)
print("Root Mean Squared Error:", rmse_lr)
print("R-squared:", r2_lr)

print("\nRandom Forest Metrics:")
print("Mean Absolute Error:", mae_rf)
print("Mean Squared Error:", mse_rf)
print("Root Mean Squared Error:", rmse_rf)
print("R-squared:", r2_rf)

print("\nGradient Boosting Metrics:")
print("Mean Absolute Error:", mae_gb)
print("Mean Squared Error:", mse_gb)
print("Root Mean Squared Error:", rmse_gb)
print("R-squared:", r2_gb)


# In[32]:


# Defining the X and the y axis
columns_to_drop = ['country', 'sum_avg_growth_last_year']
X = gdp_capita.drop(columns=columns_to_drop)
y = gdp_capita['sum_avg_growth_last_year']


# In[33]:


# Applying the LR model to 2024
LR = RandomForestRegressor()
LR.fit(X, y)
predictions_2024 = LR.predict(X)


# In[34]:


# Printing the predictions of 2024
predictions_2024


# In[35]:


# Converting the predictions of 2024 from exponencial to normal numbers and adding the new column of 2024
pred_2024 = np.round(predictions_2024, decimals=1)
gdp_capita['2024'] = pred_2024


# In[36]:


# Printing the dataset
gdp_capita


# In[37]:


# Defining the X and the y axis
columns_to_drop = ['country', '2024','sum_avg_growth_last_year']
X = gdp_capita.drop(columns=columns_to_drop)
y = gdp_capita['2024']


# In[38]:


# Applying the LR model to 2025
LR = LinearRegression()
LR.fit(X, y)
predictions_2025 = LR.predict(X)


# In[39]:


# Printing the predictions of 2024
predictions_2025


# In[40]:


# Converting the predictions of 2025 from exponencial to normal numbers and adding the new column of 2025
pred_2025 = np.round(predictions_2025, decimals=1)
gdp_capita['2025'] = pred_2025


# In[41]:


# Printing the dataset
gdp_capita


# In[42]:


# Dropping the column 'sum avg growth last year'
gdp_capita.drop('sum_avg_growth_last_year', axis=1, inplace=True)
gdp_capita


# In[43]:


# Extracting the dataset to an excel file
gdp_capita.to_excel('gdp_capita_pred_24_25.xlsx', index=False)


# In[44]:


# Plotting the top 10 and bottom of 2024
gdp_capita.sort_values(by="2024", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp_capita.head(20), gdp_capita.tail(15)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2024'], color='skyblue')


plt.xlabel('GDP per capita', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP per Capita in 2024', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}$'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('GDP_Per_Capita_2024.png')
plt.show()


# In[45]:


# Plotting the top 10 and bottom of 2025
gdp_capita.sort_values(by="2025", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp_capita.head(20), gdp_capita.tail(15)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2025'], color='skyblue')


plt.xlabel('GDP per capita', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP per Capita in 2025', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}$'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('GDP_Per_Capita_2025.png')
plt.show()


# In[46]:


#Changing the format of the table, creating a new column with all the years 
gdp_capita_pivot = pd.melt(gdp_capita, id_vars=['country'], var_name='year', value_name='gdp_capita')
gdp_capita_pivot = gdp_capita_pivot.sort_values(by='country')
gdp_capita_pivot


# In[47]:


# Sorting the dataset by country and year
gdp_capita_pivot = gdp_capita_pivot.sort_values(by=['country','year'])
gdp_capita_pivot


# In[48]:


# Extracting the dataset to an excel file
gdp_capita_pivot.to_excel('gdp_capita_pivot.xlsx', index=False)

