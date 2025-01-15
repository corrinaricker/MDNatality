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

#%%clean data
data_natality_notes = data_natality.pop('Notes').dropna()

data_income.bfill(axis=0, inplace=True)
data_income = data_income.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
list_labels = list(data_income['Label (Grouping)'])
list_new_labels = []
for i in range(0,len(list_labels)):
    if 'Maryland' in list_labels[i]:
        list_new_labels.append(list_labels[i])
        list_new_labels.append(list_labels[i] + list_labels[i+1])
        list_new_labels.append(list_labels[i] + list_labels[i+2])
        list_new_labels.append(list_labels[i] + list_labels[i+3])
        list_new_labels.append(list_labels[i] + list_labels[i+4])


#%%EDA
#data.info()
describe = data_natality.describe()

'''to do:
    -add low, normal, high to birth weight 
    -same for gestational age
    -look up census data for MD for income, etc by county
    -replace 99s with Nans in prenatal visits
'''

