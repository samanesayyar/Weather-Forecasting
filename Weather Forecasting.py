# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:54:32 2024

@author: Samane
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.renderers.default='browser'
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

data=pd.read_csv('Dataset/DailyDelhiClimateTrain.csv')

#print(data.head())
#print(data.isnull().sum())
##  the descriptive statistics ##
#print(data.describe())
#print(data.info())
##Mean Temperature in Delhi Over the Years##
#fig = px.line(data,x='date',y='meantemp',title="Mean Temperature in Delhi Over the Years")
#fig.show()
##Relationship Between Temperature and Humidity##
#fig2 = px.scatter(data,x='humidity',y='meantemp',size='meantemp',trendline='ols',title='Relationship Between Temperature and Humidity')
#fig2.show()

##convert the data type of the date column into datetime##
data['date']= pd.to_datetime(data['date'], format='%Y-%m-%d')
data['year']= data['date'].dt.year
data['month']=data['date'].dt.month

plt.style.use('dark_background')
plt.figure(figsize=(15,10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data, x="month", y="meantemp", hue='year')
plt.show()

## rename date and meantemp column name ##
forcast_data= data.rename(columns={"date":"ds","meantemp":"y"})
print(forcast_data)


##Forecasting Weather using Python##
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

model=Prophet()
model.fit(forcast_data)
forcasts=model.make_future_dataframe(periods=365)
predictions=model.predict(forcasts)
fig3=plot_plotly(model,predictions)
fig3.show()
