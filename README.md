# Case Study: Bellabeat Customer Behavior Analysis

### Author: Patricia Louis
### Date: 
---
This project aims to perform descriptive analysis on a FitBit dataset that will provide insights used to make recommendations to the marketing department of the fictional Bellabeat company.

## Background 

## Ask
### Business Task:
### Stakeholders:

## Prepare
### Data Source
### ROCCC
### Limitations
## Process
### Data Cleaning
Importing necessary Python libraries.
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
!pip install skimpy
import skimpy # alternative to pyjanitor
```
Loading CSV files.
``` python
daily_activity = pd.read_csv('dailyActivity_merged copy.csv')
daily_intensities = pd.read_csv('dailyIntensities_merged copy.csv')
hourly_intensities = pd.read_csv('hourlyIntensities_revised.csv')
hourly_calories = pd.read_csv('hourlyCalories_revised.csv')
hourly_steps = pd.read_csv('hourlySteps_revised.csv')
sleep_day = pd.read_csv('sleepDay_revised.csv')
weight_log = pd.read_csv('weightLogInfo_revised.csv')
```
Separating 'activity_hour' columns into 'hour' and 'weekday' columns.
``` python
hourly_calories[['date','hour']] = hourly_calories.ActivityHour.str.split(expand=True)
hourly_steps[['date','hour']] = hourly_steps.ActivityHour.str.split(expand=True)
hourly_intensities[['date','hour']] = hourly_intensities.ActivityHour.str.split(expand=True)
weight_log[['date','hour']] = weight_log.Date.str.split(expand=True)

# Deleting ActivityHour column
hourly_calories.drop(columns = ['ActivityHour'], inplace = True)
hourly_steps.drop(columns = ['ActivityHour'], inplace = True)
hourly_intensities.drop(columns = ['ActivityHour'], inplace = True)
weight_log.drop(columns = ['Date'], inplace = True)
hourly_intensities.head()

# Reordering columns
hourly_calories = hourly_calories[['Id', 'date', 'hour', 'Calories']]
hourly_steps = hourly_steps[['Id', 'date', 'hour', 'StepTotal']]
hourly_intensities = hourly_intensities[['Id', 'date', 'hour', 'TotalIntensity', 'AverageIntensity']]
weight_log = weight_log[['Id', 'date', 'hour', 'WeightKg', 'WeightPounds', 'Fat', 'BMI', 'IsManualReport', 'LogId']]
```
Verifying columns are of the proper data type.
``` python
daily_activity['Id'] = daily_activity['Id'].astype(str)
daily_activity['ActivityDate'] = pd.to_datetime(daily_activity['ActivityDate'], format = '%m/%d/%Y')
daily_activity.dtypes

hourly_intensities['Id'] = hourly_intensities['Id'].astype(str)
# Casting 'date' and 'hour' columns as datetime and time objects
hourly_intensities['date'] = pd.to_datetime(hourly_intensities['date'], format = '%m/%d/%y')
hourly_intensities['hour'] = pd.to_datetime(hourly_intensities['hour'], format = '%H:%M').dt.time
hourly_intensities.dtypes

hourly_calories['Id'] = hourly_calories['Id'].astype(str)
# Casting 'date' and 'hour' columns as datetime and time objects
hourly_calories['date'] = pd.to_datetime(hourly_calories['date'], format = '%m/%d/%y')
hourly_calories['hour'] = pd.to_datetime(hourly_calories['hour'], format = '%H:%M').dt.time
hourly_calories.dtypes

hourly_steps['Id'] = hourly_steps['Id'].astype(str)
# Casting 'date' and 'hour' columns as datetime and time objects
hourly_steps['date'] = pd.to_datetime(hourly_steps['date'], format = '%m/%d/%y')
hourly_steps['hour'] = pd.to_datetime(hourly_steps['hour'], format = '%H:%M').dt.time
hourly_steps.dtypes

sleep_day['Id'] = sleep_day['Id'].astype(str)
sleep_day['SleepDay'] = pd.to_datetime(sleep_day['SleepDay'], format = '%m/%d/%y %H:%M')
sleep_day.dtypes

weight_log['Id'] = weight_log['Id'].astype(str)
weight_log['LogId'] = weight_log['LogId'].astype(str)
# Casting 'date' and 'hour' columns as datetime and time objects
weight_log['date'] = pd.to_datetime(weight_log['date'], format = '%m/%d/%y')
weight_log['hour'] = pd.to_datetime(weight_log['hour'], format = '%H:%M').dt.time
weight_log.dtypes
```
Checking for null values.
``` python
print(daily_activity.isnull().values.any())
print(daily_intensities.isnull().values.any())
print(hourly_intensities.isnull().values.any())
print(hourly_calories.isnull().values.any())
print(hourly_steps.isnull().values.any())
print(sleep_day.isnull().values.any())
print(weight_log.isnull().values.any())
null_rows = weight_log[weight_log.isnull().values] # Returning rows with a null value
print(null_rows.to_string())
weight_log = weight_log.drop(columns = ['Fat']) # Dropping 'Fat' columns due to the large number of missing values
```
Checking for duplicate rows.
``` python
print(daily_activity.duplicated().values.any())
print(daily_intensities.duplicated().values.any())
print(hourly_intensities.duplicated().values.any())
print(hourly_calories.duplicated().values.any())
print(hourly_steps.duplicated().values.any())
print(sleep_day.duplicated().values.any())
sleep_day.drop_duplicates(inplace=True)
print(weight_log.duplicated().values.any())
```
Changing column names to the snake_case format.
``` python
daily_activity = skimpy.clean_columns(daily_activity)
daily_activity.columns
daily_intensities = skimpy.clean_columns(daily_intensities)
daily_intensities.columns
hourly_intensities = skimpy.clean_columns(hourly_intensities)
hourly_intensities.columns
hourly_calories = skimpy.clean_columns(hourly_calories)
hourly_calories.columns
hourly_steps = skimpy.clean_columns(hourly_steps)
hourly_steps.columns
sleep_day = skimpy.clean_columns(sleep_day)
sleep_day.columns
weight_log = skimpy.clean_columns(weight_log)
weight_log.columns
```
### Data Transformation
Creating 'weekday' column in daily_activity and sleep_day data frames.
``` python
daily_activity['weekday'] = daily_activity['activity_date'].dt.day_name()
sleep_day['weekday'] = sleep_day['sleep_day'].dt.day_name()
```
Creating a dataframe that assigns an activity group of either "Low", "Moderate", "Moderately High", "High", or "Very High" based on where the user's total step count is within the quartiles.
``` python
id_activity_grouped = daily_activity.groupby('id')['total_steps'].describe()
activity_group = [] # This list will hold the activity groups of each user
for x in id_activity_grouped['mean']:
  if x <= 3789.75:
    activity_group.append('Low')
  elif x > 3789.75 and x <= 7405.50:
    activity_group.append('Moderate')
  elif x > 3789.75 and x <= 7405.50:
    activity_group.append('Moderately High')
  elif x > 7405.50 and x <= 10727.00:
    activity_group.append('High')
  elif x > 10727.00 and x <= 39019.00:
    activity_group.append('Very High')

user_activity = {
    'id': id_activity_grouped.index,
    'activity_group': activity_group
}
user_activity = pd.DataFrame(user_activity)
user_activity.head()
```
Joining dataframes with user_activity dataframe.
``` python
daily_activity = pd.merge(daily_activity, user_activity, how='inner', on='id')
sleep_day = pd.merge(sleep_day, user_activity, how='inner', on='id')
weight_log = pd.merge(weight_log, user_activity, how='inner', on='id')
hourly_calories = pd.merge(hourly_calories, user_activity, how='inner', on='id')
hourly_steps = pd.merge(hourly_steps, user_activity, how='inner', on='id')
hourly_intensities = pd.merge(hourly_intensities, user_activity, how='inner', on='id')
```
## Analyze

## Share

## Act

