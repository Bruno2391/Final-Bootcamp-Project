#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
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


#Import dataset
gdp_ppp = pd.read_csv(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets\gdp_ppp.csv")
gdp_ppp


# In[3]:


# Rename columns and lower letters
cols = []
for i in range(len(gdp_ppp.columns)):
    cols.append(gdp_ppp.columns[i].lower().replace(' ', ''))
gdp_ppp.columns = cols
gdp_ppp


# In[4]:


# Checking the number of rows and features
gdp_ppp.shape


# In[5]:


# Dropping the column 'country code'
gdp_ppp = gdp_ppp.drop(columns=['countrycode'], axis=1)
gdp_ppp


# In[6]:


# Checking a summary of the df to understand the structure and content of the dataset, as nul values, and data types
gdp_ppp.info()


# In[7]:


# Display option for pandas to show all rows of the dataset
pd.set_option('display.max_rows', None)
print(gdp_ppp)


# In[8]:


#Dropping rows that are not relevants
indexes_to_eliminate = [4,32,44,55,56,57,58,59,68,92,96,97,98,99,101,104,122,128,129,130,133,134,185,187,191,192,209,211,212,136,147,150,155,164,175,177,224,225,230,232,234,235,243,253]
gdp_ppp.drop(indexes_to_eliminate, inplace=True)
gdp_ppp


# In[9]:


#Checking the columns of the dataset
gdp_ppp.columns


# In[10]:


#Checking the sum of the null values of each feature 
gdp_ppp.isna().sum()


# In[11]:


#Checking duplicate data of the dataset
gdp_ppp.duplicated().sum()


# In[12]:


#Filling up the missing values with the mean, and also others with 0 «, at end im doing the reset of the index 
gdp_ppp.set_index('country', inplace=True)
gdp_ppp = gdp_ppp.apply(lambda row: row.fillna(row.mean()), axis=1)
gdp_ppp = gdp_ppp.fillna(0)
gdp_ppp.reset_index(inplace=True)
gdp_ppp


# In[13]:


#Adding new columns 2020, 2021,2022 and 2023
gdp_ppp['2020'] = 0.0
gdp_ppp['2021'] = 0.0
gdp_ppp['2022'] = 0.0
gdp_ppp['2023'] = 0.0


# In[14]:


# Printing the dataset
gdp_ppp


# In[15]:


# Filling up the new columns with the values of the gdp ppp, where the code do the calculation of the variance 1980 until 2019.
variance_df = gdp_ppp.loc[:, '1990':'2018'].var(axis=1)
for year in range(2019, 2024):
       gdp_ppp[str(year)] = gdp_ppp.apply(lambda row: row[str(year-1)] + (row[str(year-1)] - row['1990']) / row['1990']
                                          if row['1990'] != 0
                                          else 0, axis=1)


# In[16]:


gdp_ppp


# In[17]:


#Checking some statistics 
gdp_ppp.describe().T


# In[18]:


# Plotting the GDP trends for different countries
for index, row in gdp_ppp.iterrows():
    country_name = row['country']
    gdp_data = row.values[1:]

    plt.figure(figsize=(8, 6))
    plt.plot(gdp_ppp.columns[1:], gdp_data)
    plt.title(f'GDP Trend for {country_name}')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.xticks(rotation='vertical')
    plt.show()


# In[19]:


# Plotting GDP trends for all the countries
plt.figure(figsize=(15, 8)) 

for index, row in gdp_ppp.iterrows():
    country_name = row['country']
    gdp_data = row.values[1:]

    plt.plot(gdp_ppp.columns[1:], gdp_data, label=country_name)

plt.title('GDP PPP Trends for All Countries')
plt.xlabel('Year')
plt.ylabel('GDP PPP')
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.tight_layout()
plt.show()


# In[20]:


gdp_ppp


# In[21]:


#Extracting as an excel file 
gdp_ppp.to_excel('gdp_ppp_clean.xlsx', index=False)


# In[22]:


#Creating a new column with the average of gdp fro the last 10 years
gdp_ppp['Avg_last_10_years'] = gdp_ppp[['2013','2014','2015','2016','2017','2018','2019','2020','2021', '2022', '2023']].mean(axis=1)
gdp_ppp


# In[23]:


# Defining the X and the y axis
columns_to_drop = ['country', 'Avg_last_10_years']
X = gdp_ppp.drop(columns=columns_to_drop)
y = gdp_ppp['Avg_last_10_years']


# In[24]:


# Spliting the x and y 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


#Testing the LR model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)


# In[26]:


# Printing the predictions
predictions_lr


# In[27]:


#Testing the RF model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)


# In[28]:


# Printing the predictions
predictions_rf


# In[29]:


#Testing the GB model
from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)


# In[30]:


# Printing the predictions
predictions_gb


# In[31]:


# Evaluation of the LR Model 
mae_lr = mean_absolute_error(y_test, predictions_lr)
mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)


# In[32]:


# Evaluation of the RF Model 
mae_rf = mean_absolute_error(y_test, predictions_rf)
mse_rf = mean_squared_error(y_test, predictions_rf)
r2_rf = r2_score(y_test, predictions_rf)


# In[33]:


# Evaluation of the GB Model 
mae_gb = mean_absolute_error(y_test, predictions_gb)
mse_gb = mean_squared_error(y_test, predictions_gb)
r2_gb = r2_score(y_test, predictions_gb)


# In[34]:


#  Printing the evaluations of the models 
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


# In[35]:


#Defining the X and y axis to 2024
columns_to_drop = ['country', 'Avg_last_10_years']
X = gdp_ppp.drop(columns=columns_to_drop)
y = gdp_ppp['Avg_last_10_years']


# In[36]:


#Applying the LR model to 2024
LR= LinearRegression()
LR.fit(X, y)
predictions_2024 = LR.predict(X)


# In[37]:


#Printing the predictions of 2024
predictions_2024


# In[38]:


# Converting the values from exponencial to normal numbers
pred_2024 = np.round(predictions_2024, decimals=1)
gdp_ppp['2024'] = pred_2024


# In[39]:


# Printing the dataset
gdp_ppp


# In[40]:


#Defining the X and y axis to 2025
columns_to_drop = ['country', 'Avg_last_10_years','2024']
X = gdp_ppp.drop(columns=columns_to_drop)
y = gdp_ppp['2024']


# In[41]:


#Applying the model LR to 2025
LR= LinearRegression()
LR.fit(X, y)
predictions_2025 = LR.predict(X)


# In[42]:


#Printing the predictions of 2025 
predictions_2025


# In[43]:


#Converting the predictions from exponencial to normal numbers
pred_2025 = np.round(predictions_2025, decimals=1)
gdp_ppp['2025'] = pred_2025


# In[44]:


#Printing the dataset 
gdp_ppp


# In[45]:


#DRoppig the column ' avg last 10 years'
gdp_ppp.drop('Avg_last_10_years', axis=1, inplace=True)
gdp_ppp


# In[46]:


#Extracting the dataset to excel 
gdp_ppp.to_excel('gdp_ppp_pred_24_25.xlsx', index=False)


# In[47]:


#Plotting the top 10 and bottom of 2024
gdp_ppp.sort_values(by="2024", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp_ppp.head(20), gdp_ppp.tail(15)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2024'], color='skyblue')


plt.xlabel('GDP PPP', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP PPP in 2024', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}$'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()  
plt.show()


# In[48]:


#Plotting the top 10 and bottom of 2025
gdp_ppp.sort_values(by="2025", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp_ppp.head(20), gdp_ppp.tail(15)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2025'], color='skyblue')


plt.xlabel('GDP PPP', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP PPP in 2025', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}$'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('GDP_PPP_2025.png')
plt.show()


# In[49]:


#Changing the format of the table, creating a new column with all the years 
gdp_ppp_pivot = pd.melt(gdp_ppp, id_vars=['country'], var_name='year', value_name='gdp')
gdp_ppp_pivot = gdp_ppp_pivot.sort_values(by='country')
gdp_ppp_pivot


# In[50]:


# Sorting the values of the dataset by country and after by year
gdp_ppp_pivot = gdp_ppp_pivot.sort_values(by=['country','year'])
gdp_ppp_pivot


# In[51]:


#Extracting the dataset to excel file 
gdp_ppp_pivot.to_excel('gdp_ppp_pivot.xlsx', index=False)

