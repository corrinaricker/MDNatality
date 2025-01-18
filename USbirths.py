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

#%%load data
data_natality = pd.read_csv('C:/Users/crick/Documents/ML practice/Natality2016-2023.txt',
                            sep='\t')
data_income = pd.read_csv('C:/Users/crick/Documents/ML practice/MD_census_income.csv')

#%%clean data and add some features
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


#%%EDA
#data.info()
describe = data_natality.describe()

'''to do:
    -add low, normal, high to birth weight 
    -same for gestational age
    -look up census data for MD for income, etc by county
    -replace 99s with Nans in prenatal visits
'''

