#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
import geopandas as gpd
import math
import matplotlib.gridspec as gridspec


# In[2]:


# Import dataset 
gdp = pd.read_csv(r"C:\Users\btdjf\Desktop\Ironhack 2\Final Bootcamp Project\Final-Bootcamp-Project\Datasets\countries of the world.csv")
gdp


# In[3]:


# Checking a summary of the df to understand the structure and content of the dataset, as nul values, and data types
gdp.info()


# In[4]:


# Display option to show all rows of the dataset
pd.set_option('display.max_rows', None)
print(gdp)


# In[5]:


# Rename columns to lower letters, and replace spaces
cols = []
for i in range(len(gdp.columns)):
    cols.append(gdp.columns[i].lower().replace(' ', ''))
gdp.columns = cols
gdp


# In[6]:


# Changing the name of the features and taking off special characters
gdp.columns = (["country","region","population","area","density","coastline_area_ratio","net_migration","infant_mortality","gdp_per_capita",
                  "literacy","phones","arable","crops","other","climate","birthrate","deathrate","agriculture","industry",
                  "service"])
gdp


# In[7]:


# Checking the columns of the dataset
gdp.columns


# In[8]:


# Checking the unique values for the country column
gdp['country'].unique()


# In[9]:


# Count of the number of ocurrences in the column country
gdp['country'].value_counts()


# In[10]:


# Checking the unique values for the region column
gdp['region'].unique()


# In[11]:


# Count of the number of ocurrences in the column region
gdp['region'].value_counts()


# In[12]:


# Checking the unique values for the population column
gdp['population'].unique()


# In[13]:


# Count of the number of ocurrences in the column population
gdp['population'].value_counts()


# In[14]:


# Checking the unique values for the density column
gdp['density'].unique()


# In[15]:


# Count of the number of ocurrences in the column density
gdp['density'].value_counts()


# In[16]:


# Checking the unique values for the coastline ratio column
gdp['coastline_area_ratio'].unique()


# In[17]:


# Count of the number of ocurrences in the column coastline ratio
gdp['coastline_area_ratio'].value_counts()


# In[18]:


# Checking the unique values for the net migration column
gdp['net_migration'].unique()


# In[19]:


# Count of the number of ocurrences in the column net migration
gdp['net_migration'].value_counts()


# In[20]:


# Checking the unique values for the infant mortality column
gdp['infant_mortality'].unique()


# In[21]:


# Count of the number of ocurrences in the column infant mortality
gdp['infant_mortality'].value_counts()


# In[22]:


# Checking the unique values for the gdp per capita column
gdp['gdp_per_capita'].unique()


# In[23]:


# Count of the number of ocurrences in the column gdp per capita
gdp['gdp_per_capita'].value_counts()


# In[24]:


# Checking the unique values for the literacy column
gdp['literacy'].unique()


# In[25]:


# Count of the number of ocurrences in the column literacy
gdp['literacy'].value_counts()


# In[26]:


# Checking the unique values for the phones column
gdp['phones'].unique()


# In[27]:


# Count of the number of ocurrences in the column phones
gdp['phones'].value_counts()


# In[28]:


# Checking the unique values for the arable column
gdp['arable'].unique()


# In[29]:


# Count of the number of ocurrences in the column arable
gdp['arable'].value_counts()


# In[30]:


# Checking the unique values for the crops column
gdp['crops'].unique()


# In[31]:


# Count of the number of ocurrences in the column crops
gdp['crops'].value_counts()


# In[32]:


# Checking the unique values for the other column
gdp['other'].unique()


# In[33]:


# Count of the number of ocurrences in the column other
gdp['other'].value_counts()


# In[34]:


# Checking the unique values for the climate column
gdp['climate'].unique()


# In[35]:


# Count of the number of ocurrences in the column climate
gdp['climate'].value_counts()


# In[36]:


# Checking the unique values for the birthrate column
gdp['birthrate'].unique()


# In[37]:


# Count of the number of ocurrences in the column birthrate
gdp['birthrate'].value_counts()


# In[38]:


# Checking the unique values for the deathrate column
gdp['deathrate'].unique()


# In[39]:


# Count of the number of ocurrences in the column deathrate
gdp['deathrate'].value_counts()


# In[40]:


# Checking the unique values for the agriculture column
gdp['agriculture'].unique()


# In[41]:


# Count of the number of ocurrences in the column agriculture
gdp['agriculture'].value_counts()


# In[42]:


# Checking the unique values for the industry column
gdp['industry'].unique()


# In[43]:


# Count of the number of ocurrences in the column industry
gdp['industry'].value_counts()


# In[44]:


# Checking the unique values for the service column
gdp['service'].unique()


# In[45]:


# Count of the number of ocurrences in the column service
gdp['service'].value_counts()


# In[46]:


# checking the struture of the dataset
gdp.shape


# In[47]:


# Checking the data types of each feature
gdp.dtypes


# In[48]:


# Checking the null values of each feature
gdp.isnull().sum()


# In[49]:


# Checking for duplicates in the dataset
gdp.duplicated().sum()


# In[50]:


#Converting columns to categorical data types: gdp.country and gdp.region columns are converted to categorical data types
#Cleaning and converting numeric columns: Columns like gdp.density, gdp.coastline_area_ratio, gdp.net_migration, 
#gdp.infant_mortality, gdp.literacy, gdp.phones, gdp.arable, gdp.crops, gdp.other, gdp.climate, gdp.birthrate, 
#gdp.deathrate, gdp.agriculture, gdp.industry, and gdp.service are being cleaned and converted to float data types.
#Converting commas to periods in the string representation of the numbers.
#Casting the resulting string to a float data type.

gdp.country = gdp.country.astype('category')

gdp.region = gdp.region.astype('category')

gdp.density = gdp.density.astype(str)
gdp.density = gdp.density.str.replace(",",".").astype(float)

gdp.coastline_area_ratio = gdp.coastline_area_ratio.astype(str)
gdp.coastline_area_ratio = gdp.coastline_area_ratio.str.replace(",",".").astype(float)

gdp.net_migration = gdp.net_migration.astype(str)
gdp.net_migration = gdp.net_migration.str.replace(",",".").astype(float)

gdp.infant_mortality = gdp.infant_mortality.astype(str)
gdp.infant_mortality = gdp.infant_mortality.str.replace(",",".").astype(float)

gdp.literacy = gdp.literacy.astype(str)
gdp.literacy = gdp.literacy.str.replace(",",".").astype(float)

gdp.phones = gdp.phones.astype(str)
gdp.phones = gdp.phones.str.replace(",",".").astype(float)

gdp.arable = gdp.arable.astype(str)
gdp.arable = gdp.arable.str.replace(",",".").astype(float)

gdp.crops = gdp.crops.astype(str)
gdp.crops = gdp.crops.str.replace(",",".").astype(float)

gdp.other = gdp.other.astype(str)
gdp.other = gdp.other.str.replace(",",".").astype(float)

gdp.climate = gdp.climate.astype(str)
gdp.climate = gdp.climate.str.replace(",",".").astype(float)

gdp.birthrate = gdp.birthrate.astype(str)
gdp.birthrate = gdp.birthrate.str.replace(",",".").astype(float)

gdp.deathrate = gdp.deathrate.astype(str)
gdp.deathrate = gdp.deathrate.str.replace(",",".").astype(float)

gdp.agriculture = gdp.agriculture.astype(str)
gdp.agriculture = gdp.agriculture.str.replace(",",".").astype(float)

gdp.industry = gdp.industry.astype(str)
gdp.industry = gdp.industry.str.replace(",",".").astype(float)

gdp.service = gdp.service.astype(str)
gdp.service = gdp.service.str.replace(",",".").astype(float)


# In[51]:


# Printing the dataset
gdp


# In[52]:


# checking the data types of the features
gdp.dtypes


# In[53]:


# Boxplots with the distribution of the features 

fig = plt.figure(figsize=(16,30))
features= ["population","area", "density", "coastline_area_ratio","net_migration","infant_mortality", "literacy", "phones", "arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"]

for i in range(len(features)):
    fig.add_subplot(9, 5, i+1)
    sns.boxplot(y=gdp[features[i]])
plt.tight_layout()
plt.show()


# In[54]:


#  Plotting the distribution of various features in the gdp
fig = plt.figure(figsize=(16,30))
features= ["population","area", "density", "coastline_area_ratio","net_migration","infant_mortality", "literacy", "phones", "arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"]

for i in range(len(features)):
    fig.add_subplot(9, 5, i+1)
    sns.distplot(gdp[features[i]])
plt.tight_layout()
plt.show()


# In[55]:


# Checking for outliers
sns.boxplot(y=gdp['gdp_per_capita'],x= gdp['region'])
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()


# In[56]:


# checking the distribution of target value gdp per capita, and outliers
fig = plt.figure(constrained_layout=True, figsize=(16,6))
grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
ax1 = fig.add_subplot(grid[0, :2])
ax1.set_title('Histogram')
sns.distplot(gdp.loc[:,'gdp_per_capita'], norm_hist=True, ax = ax1)
ax3 = fig.add_subplot(grid[:, 2])
ax3.set_title('Box Plot')
sns.boxplot(gdp.loc[:,'gdp_per_capita'], orient='v', ax = ax3)
plt.show()


# In[57]:


#Filling missing values in various columns of the 'gdp' dataframe with specific values or with the calculated median or mean
gdp['net_migration'].fillna(0, inplace=True)
gdp['infant_mortality'].fillna(0, inplace=True)
gdp['gdp_per_capita'].fillna(2500, inplace=True)
gdp['literacy'].fillna(gdp.groupby('region')['literacy'].transform('median'), inplace= True)
gdp['phones'].fillna(gdp.groupby('region')['phones'].transform('median'), inplace= True)
gdp['arable'].fillna(0, inplace=True)
gdp['crops'].fillna(0, inplace=True)
gdp['other'].fillna(0, inplace=True)
gdp['climate'].fillna(0, inplace=True)
gdp['birthrate'].fillna(gdp.groupby('region')['birthrate'].transform('mean'), inplace= True)
gdp['deathrate'].fillna(gdp.groupby('region')['deathrate'].transform('median'), inplace= True)
gdp['agriculture'].fillna(0.15, inplace=True)
gdp['service'].fillna(0.8, inplace=True)
gdp['industry'].fillna(0.05, inplace= True)


# In[58]:


# Checking for null values
gdp.isnull().sum()


# In[59]:


# Checking summary description with statistics
gdp.describe().T


# In[60]:


# Printing the dataset
gdp


# In[61]:


# Exctracting the dataset clean
gdp.to_excel('gdp_clean.xlsx', index=False)


# In[62]:


# Map graph with the gdp per country of each world
import plotly.graph_objs as go
from plotly.offline import iplot
z = dict(type='choropleth',
locations = gdp.country,
locationmode = 'country names', z = gdp.gdp_per_capita,
text = gdp.country, colorbar = {'title':'GDP per Capita'},
colorscale = 'Hot', reversescale = True)
layout = dict(title='GDP per Capita of World Countries',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [z],layout = layout)
iplot(choromap,validate=False)


# In[63]:


# Analysis of GDP per capita, net migration, and population across different regions
fig = plt.figure(figsize=(18, 24))
plt.title('Regional Analysis')
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)
sns.countplot(data= gdp, y= 'region', ax= ax1)
sns.barplot(data= gdp, y= 'region', x= 'gdp_per_capita', ax= ax2, ci= None)
sns.barplot(data= gdp, y= 'region', x= 'net_migration', ax= ax3, ci= None)
sns.barplot(data= gdp, y= 'region', x= 'population', ax= ax4, ci= None)
plt.show()


# In[64]:


# Correlation matrix to understand the correlations of the features 
correlation_matrix = gdp.drop(columns=['country','region']).corr()
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Map')
plt.show()


# In[65]:


# Checking the features of the dataset
gdp.columns


# In[66]:


# Second correlatin matrix 
correlation_matrix_2 = gdp.drop(columns=['country','region','infant_mortality','birthrate','phones','other','service']).corr()
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix_2, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Map')
plt.show()


# In[67]:


# Definition of the X and the y axis
columns_to_drop = ['country','gdp_per_capita','region']
X = gdp.drop(columns=columns_to_drop)
y = gdp['gdp_per_capita']


# In[68]:


# Train, test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[69]:


# Testing  the LR model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)


# In[70]:


# Printing the predictions
predictions_lr


# In[71]:


# Testing  the RF model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)


# In[72]:


# Printing the predictions
predictions_rf


# In[73]:


# Testing  the GB model
from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)


# In[74]:


# Printing the predictions
predictions_gb


# In[75]:


# Evaluating the LR model
mae_lr = mean_absolute_error(y_test, predictions_lr)
mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)


# In[76]:


# Evaluating the RF model
mae_rf = mean_absolute_error(y_test, predictions_rf)
mse_rf = mean_squared_error(y_test, predictions_rf)
r2_rf = r2_score(y_test, predictions_rf)


# In[77]:


# Evaluating the GB model
mae_gb = mean_absolute_error(y_test, predictions_gb)
mse_gb = mean_squared_error(y_test, predictions_gb)
r2_gb = r2_score(y_test, predictions_gb)


# In[78]:


# printing the evaluations of the models
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


# In[79]:


# Applying the GB Model
model_gb = GradientBoostingRegressor()
model_gb.fit(X, y)
predictions_gb = model_gb.predict(X)


# In[80]:


# Printing the predictions of the GB model 
predictions_gb


# In[81]:


# Adding a new column named 2024 with the predicions of 2024
gdp['2024'] = predictions_gb
gdp


# In[82]:


# Plotting the top 10 and bottom of 2023
gdp.sort_values(by="gdp_per_capita", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp.head(20), gdp.tail(15)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['gdp_per_capita'], color='skyblue')


plt.xlabel('GDP per Capita ', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP per Capita in 2023', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}$'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('GDP_2023.png')
plt.show()


# In[83]:


# Plotting the top 10 and bottom of 2024
gdp.sort_values(by="2024", ascending=False, inplace=True)
top_bottom_10 = pd.concat([gdp.head(20), gdp.tail(15)])
plt.figure(figsize=(12, 10))
bars = plt.barh(top_bottom_10['country'], top_bottom_10['2024'], color='skyblue')


plt.xlabel('GDP per Capita ', fontsize=14)
plt.title('Top and Bottom 10 Countries by GDP per Capita in 2024', fontsize=16)
plt.gca().invert_yaxis()  


plt.yticks(fontsize=10, rotation=45)


for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, '{:.1f}$'.format(width), ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('GDP_Pred_2024.png')
plt.show()


# In[84]:


# Extracting the dataset with the predictions of 2024
gdp.to_excel('gdp_pred.xlsx', index=False)

