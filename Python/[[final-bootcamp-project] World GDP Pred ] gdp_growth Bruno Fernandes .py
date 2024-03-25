#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries 
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import geopandas as gpd
import math
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


#Importing dataset
gdp_growth = pd.read_csv(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets\gdp_growth.csv", encoding='ISO-8859-1')
gdp_growth


# In[3]:


# Rename column 'country_name to country'
gdp_growth.rename(columns={'country_name': 'Country'}, inplace=True)
gdp_growth


# In[4]:


# Rename columns names to lower case and replacing spaces
cols = []
for i in range(len(gdp_growth.columns)):
    cols.append(gdp_growth.columns[i].lower().replace(' ', ''))
gdp_growth.columns = cols
gdp_growth


# In[5]:


# Checking the structure of the dataset
gdp_growth.shape


# In[6]:


#Dropping the 'indicator name' column 
gdp_growth = gdp_growth.drop(columns=['indicator_name'], axis=1)
gdp_growth


# In[7]:


# Checking a summary of the df to understand the structure and content of the dataset, as nul values, and data types 
gdp_growth.info()


# In[8]:


#Checking all the rows of the dataset
pd.set_option('display.max_rows', None)
print(gdp_growth)


# In[9]:


#checking the sum of null values 
gdp_growth.isna().sum()


# In[10]:


#Checking the duplicates in the dataset
gdp_growth.duplicated().sum()


# In[11]:


#Filling up the null values with 0
gdp_growth.fillna(0.0, inplace=True)
gdp_growth


# In[12]:


# Checking statistics of the dataset
gdp_growth.describe().T


# In[13]:


#checkinh the columns of the dataset
gdp_growth.columns


# In[14]:


#checkinh the unique values of each feature
gdp_growth.nunique()


# In[15]:


#Plotting the trend of the gdp of each country 
for index, row in gdp_growth.iterrows():
    country_name = row['country']
    gdp_data = row.values[1:]

    plt.figure(figsize=(8, 6))
    plt.plot(gdp_growth.columns[1:], gdp_data)
    plt.title(f'GDP Growth {country_name}')
    plt.xlabel('Year')
    plt.ylabel('Growth(%)')
    plt.xticks(rotation='vertical')
    plt.show()


# In[16]:


#Plotting hte trend of the gdp of all the countries
plt.figure(figsize=(15, 9)) 

for index, row in gdp_growth.iterrows():
    country_name = row['country']
    gdp_data = row.values[1:]

    plt.plot(gdp_growth.columns[1:], gdp_data, label=country_name)

plt.title('GDP Growth All Countries')
plt.xlabel('Year')
plt.ylabel('Growth')
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()


# In[17]:


#Printing the dataset 
gdp_growth


# In[18]:


# Extracting the dataset to excel file 
gdp_growth.to_excel('gdp_growth_clean.xlsx', index=False)


# In[19]:


# Creating a new column with the mean of each country 
avg_growth = gdp_growth.iloc[:, 1:].mean(axis=1) 
gdp_growth['avg_growth'] = avg_growth
gdp_growth.round(1)


# In[20]:


# Putting the values in decimals rounds
gdp_growth = gdp_growth.round(1)
gdp_growth


# In[21]:


#Defining the X and the y axis
columns_to_drop = ['country','avg_growth']
X = gdp_growth.drop(columns=columns_to_drop)
y = gdp_growth['avg_growth']


# In[22]:


# Spliting the X and y in train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[23]:


# Testing the model LR 
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)


# In[24]:


# Printing the predictions 
predictions_lr


# In[25]:


# Testing the model RF 
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)


# In[26]:


# Printing the predictions 
predictions_rf


# In[27]:


# Testing the model GB 
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)


# In[28]:


# Printing the predictions 
predictions_gb


# In[29]:


# Evaluating the model LR
mae_lr = mean_absolute_error(y_test, predictions_lr)
mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)


# In[30]:


# Evaluating the model RF
mae_rf = mean_absolute_error(y_test, predictions_rf)
mse_rf = mean_squared_error(y_test, predictions_rf)
r2_rf = r2_score(y_test, predictions_rf)


# In[31]:


# Evaluating the model GB
mae_gb = mean_absolute_error(y_test, predictions_gb)
mse_gb = mean_squared_error(y_test, predictions_gb)
r2_gb = r2_score(y_test, predictions_gb)


# In[32]:


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


# In[33]:


#Defining the X and y axis 
columns_to_drop = ['country','avg_growth']
X = gdp_growth.drop(columns=columns_to_drop)
y = gdp_growth['avg_growth']


# In[34]:


# Applying the LR model
LR= LinearRegression()
LR.fit(X, y)
predictions_2025 = LR.predict(X)


# In[35]:


# Printing the predictions of 2025
predictions_2025


# In[36]:


# Converting the results from exponencial numbers to normal numbers 
pred_2025 = np.round(predictions_2025, decimals=1)
gdp_growth['2025'] = pred_2025


# In[38]:


# Printing the dataset
gdp_growth


# In[39]:


#Dropping the column 'avg growth'
gdp_growth.drop(columns=['avg_growth'], inplace=True)
gdp_growth


# In[40]:


# Extracting the dataset to excel file 
gdp_growth.to_excel('gdp_growth_pred_24_25.xlsx', index=False)


# In[41]:


#Plotting the top 10 and bottoms of 2024 

gdp_growth.sort_values(by="2024", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp_growth.head(10), gdp_growth.tail(10)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2024'], color='skyblue')


plt.xlabel('Average Growth Rate (%)', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP Growth Rate in 2024', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}%'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout() 
plt.savefig('GDP_Growth_Rate_2024.png')
plt.show()


# In[42]:


#Plotting the top 10 and bottoms of 2025
gdp_growth.sort_values(by="2025", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp_growth.head(10), gdp_growth.tail(10)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2025'], color='skyblue')


plt.xlabel('Average Growth Rate (%)', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP Growth Rate in 2025', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}%'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('GDP_Growth_Rate_2025.png')
plt.show()


# In[43]:


# Crewating a new column with the years and sorting the values by country
gdp_growth_pivot = pd.melt(gdp_growth, id_vars=['country'], var_name='year', value_name='gdp_growth')
gdp_growth_pivot = gdp_growth_pivot.sort_values(by='country')
gdp_growth_pivot


# In[44]:


# Sorting the values by country and year
gdp_growth_pivot = gdp_growth_pivot.sort_values(by=['country','year'])
gdp_growth_pivot


# In[45]:


# Extracting the dataset to an excel file 
gdp_growth_pivot.to_excel('gdp_growth_pivot.xlsx', index=False)

