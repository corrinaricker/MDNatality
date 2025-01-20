# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:07:16 2025

@author: crick
"""

#%%load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

#%%load data
data_natality = pd.read_csv('C:/Users/crick/Documents/ML practice/Natality2016-2023.txt',
                            sep='\t')
data_income = pd.read_csv('C:/Users/crick/Documents/ML practice/MD_census_income.csv')
#%%clean data
data_natality_notes = data_natality.pop('Notes').dropna()

#the data has some formatting related to nested categories
#backfill the NaNs related to this and strip the whitespace from the nesting
data_income.bfill(axis=0, inplace=True)
data_income = data_income.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data_income = data_income[data_income['Label (Grouping)'] != 'Estimate']

#add the county and state names to their sub-categories
list_labels = list(data_income['Label (Grouping)']) #this is a list of the current labels
list_new_labels = [] #this will hold the new labels
for i in range(0,len(list_labels)): #go through each item in the list
    if 'Maryland' in list_labels[i]: 
        #if the item has "Maryland in the name, it will be followed by 
        #four sub-categories. Add the county name to the new list and
        #then for each sub-cat, add county name + sub-cat name to new list
        list_new_labels.append(list_labels[i])
        list_new_labels.append(list_labels[i] + ' ' + list_labels[i+1])
        list_new_labels.append(list_labels[i] + ' ' + list_labels[i+2])
        list_new_labels.append(list_labels[i] + ' ' + list_labels[i+3])
        list_new_labels.append(list_labels[i] + ' ' + list_labels[i+4])

#make a column for county using the list_new_labels
list_counties = [] 
for i in range(0,len(list_new_labels)):
    county = list_new_labels[i].split(',')[0]
    list_counties.append(county)
data_income['County'] = list_counties
#change the values for Maryland totals in the County columns
data_income.loc[data_income['County'].str.startswith('Maryland'), 'County'] = 'Maryland Total'
#drop some columns
cols_to_drop = [col for col in data_income.columns if col.startswith('PERCENT')]
data_income.drop(cols_to_drop, axis=1, inplace=True)

#%%check for mising values
rows = data_natality.shape[0]
print('Counties missing ', 
      data_natality['County of Residence'].isnull().sum(), 
      '{:.1%}'.format(data_natality['County of Residence'].isnull().sum()/rows))
print('Prenatal Visits missing ', 
      data_natality['Number of Prenatal Visits'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Number of Prenatal Visits'].isnull().sum()/rows))
print('Age of Mother missing ', 
      data_natality['Age of Mother 9'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Age of Mother 9'].isnull().sum()/rows))
print('Mother Education missing ', 
      data_natality["Mother's Education"].isnull().sum(), 
      '{:.1%}'.format(data_natality["Mother's Education"].isnull().sum()/rows))
print('NICU Admission missing ', 
      data_natality['NICU Admission'].isnull().sum(), 
      '{:.1%}'.format(data_natality['NICU Admission'].isnull().sum()/rows))
print('Births missing ', 
      data_natality['Births'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Births'].isnull().sum()/rows))
print('Gestational Age missing ', 
      data_natality['Average OE Gestational Age (weeks)'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average OE Gestational Age (weeks)'].isnull().sum()/rows))
print('Birth Weight missing ', 
      data_natality['Average Birth Weight (grams)'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average Birth Weight (grams)'].isnull().sum()/rows))
print('BMI missing ', 
      data_natality['Average Pre-pregnancy BMI'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average Pre-pregnancy BMI'].isnull().sum()/rows))
print('Birth Interval missing ', 
      data_natality['Average Interval Since Last Live Birth (months)'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average Interval Since Last Live Birth (months)'].isnull().sum()/rows))
#all features have the same number missing/null and it's a very small
#percentage of the total (less than 1%) so I will drop nans
data_natality = data_natality.dropna()
#there are some "Not Applicable" values in the Prenatal Visits column
#and in the Intervale Since Last Live Birth column
#check to see how mandy
print('Prenatal Visits labeled Not Applicable ', 
      data_natality.loc[data_natality['Average Number of Prenatal Visits']=='Not Applicable']['Average Number of Prenatal Visits'].count(),
      '{:.1%}'.format(data_natality.loc[data_natality['Average Number of Prenatal Visits']=='Not Applicable']['Average Number of Prenatal Visits'].count()/rows))
print('Live Birth labeled Not Applicable ', 
      data_natality.loc[data_natality['Average Interval Since Last Live Birth (months)']=='Not Applicable']['Average Interval Since Last Live Birth (months)'].count(),
      '{:.1%}'.format(data_natality.loc[data_natality['Average Interval Since Last Live Birth (months)']=='Not Applicable']['Average Interval Since Last Live Birth (months)'].count()/rows))

#5% and 1.2%, acceptable to drop
data_natality = data_natality.loc[data_natality['Average Number of Prenatal Visits']!='Not Applicable']
data_natality = data_natality.loc[data_natality['Average Interval Since Last Live Birth (months)']!='Not Applicable']
#change the datatypes of these columns to floats
data_natality['Average Number of Prenatal Visits'] = data_natality['Average Number of Prenatal Visits'].astype(float)
data_natality['Average Interval Since Last Live Birth (months)'] = data_natality['Average Interval Since Last Live Birth (months)'].astype(float)


#%%set up data for modelling
#convet qualitative values to numeric
#age of mother should be ordinal
ord_encoder = OrdinalEncoder(categories=[['15-19', '20-24', '25-29', '30-34', 
                                          '35-39', '40-44', '45-49']])
data_natality['Age of Mother encoded'] = ord_encoder.fit_transform(data_natality[['Age of Mother 9 Code']])
 
#get standard deviations to get low, avg, and high for interval since last live birth
#apply BMI categories
#use standard deviations to get low, medium, and high birth weights
birth_weight_mean = data_natality['Average Birth Weight (grams)'].mean()
birth_weight_stdev = data_natality['Average Birth Weight (grams)'].std()
birth_weight_low_threshold = birth_weight_mean - birth_weight_stdev
birth_weight_high_threshold = birth_weight_mean + birth_weight_stdev
def birth_weight_thresholds(bw):
    if bw < birth_weight_low_threshold:
        return 1
    elif bw > birth_weight_high_threshold:
        return 3
    else:
        return 2
data_natality['Birth Weight Category'] = data_natality['Average Birth Weight (grams)'].apply(birth_weight_thresholds)
#use standard deviations to get low, medium, and high intervals since last live birth
interval_mean = data_natality['Average Interval Since Last Live Birth (months)'].mean()
interval_stdev = data_natality['Average Interval Since Last Live Birth (months)'].std()
interval_low_threshold = interval_mean - interval_stdev
interval_high_threshold = interval_mean + interval_stdev
def interval_thresholds(i):
    if i < interval_low_threshold:
        return 1
    elif i > interval_high_threshold:
        return 3
    else:
        return 2
data_natality['Interval Category'] = data_natality['Average Interval Since Last Live Birth (months)'].apply(interval_thresholds)

    

#%%EDA
#data.info()
describe = data_natality.describe()

numerical_columns = data_natality.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(data_natality[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(data_natality[feature].skew(), 2)}")

plt.tight_layout()
plt.show()

'''to do:
    -add low, normal, high to birth weight 
    -same for gestational age
    -look up census data for MD for income, etc by county
    -replace 99s with Nans in prenatal visits
'''

