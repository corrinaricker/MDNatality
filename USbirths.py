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
#%%clean data and merge
data_natality_notes = data_natality.pop('Notes').dropna()
data_natality['County of Residence'] = data_natality['County of Residence'].str.strip(', MD')

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
data_income_totals = data_income.loc[data_income['Label (Grouping)'].str.endswith('Maryland')]
data_income_totals['Median income (dollars)'] = data_income_totals['Median income (dollars)'].str.replace(',', '').astype(float)
data_income_totals['Mean income (dollars)'] = data_income_totals['Mean income (dollars)'].str.replace(',', '').astype(float)


data_natality = data_natality.merge(data_income_totals[['County', 'Median income (dollars)', 'Mean income (dollars)']],
                                    how='left', left_on='County of Residence',
                                    right_on='County')

#%%check for mising values
rows = data_natality.shape[0]
print('Counties missing', 
      data_natality['County of Residence'].isnull().sum(), 
      '{:.1%}'.format(data_natality['County of Residence'].isnull().sum()/rows))
print('Prenatal Visits missing', 
      data_natality['Number of Prenatal Visits'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Number of Prenatal Visits'].isnull().sum()/rows))
print('Age of Mother missing', 
      data_natality['Age of Mother 9'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Age of Mother 9'].isnull().sum()/rows))
print('Mother Education missing', 
      data_natality["Mother's Education"].isnull().sum(), 
      '{:.1%}'.format(data_natality["Mother's Education"].isnull().sum()/rows))
print('NICU Admission missing', 
      data_natality['NICU Admission'].isnull().sum(), 
      '{:.1%}'.format(data_natality['NICU Admission'].isnull().sum()/rows))
print('Births missing', 
      data_natality['Births'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Births'].isnull().sum()/rows))
print('Gestational Age missing', 
      data_natality['Average OE Gestational Age (weeks)'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average OE Gestational Age (weeks)'].isnull().sum()/rows))
print('Birth Weight missing', 
      data_natality['Average Birth Weight (grams)'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average Birth Weight (grams)'].isnull().sum()/rows))
print('BMI missing', 
      data_natality['Average Pre-pregnancy BMI'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average Pre-pregnancy BMI'].isnull().sum()/rows))
print('Birth Interval missing', 
      data_natality['Average Interval Since Last Live Birth (months)'].isnull().sum(), 
      '{:.1%}'.format(data_natality['Average Interval Since Last Live Birth (months)'].isnull().sum()/rows))
#all features have the same number missing/null and it's a very small
#percentage of the total (less than 1%) so I will drop nans
data_natality = data_natality.dropna()
#there are some "Not Applicable" values in the Prenatal Visits column
#and in the Intervale Since Last Live Birth column
#check to see how mandy
print('Prenatal Visits labeled Not Applicable', 
      data_natality.loc[data_natality['Average Number of Prenatal Visits']=='Not Applicable']['Average Number of Prenatal Visits'].count(),
      '{:.1%}'.format(data_natality.loc[data_natality['Average Number of Prenatal Visits']=='Not Applicable']['Average Number of Prenatal Visits'].count()/rows))
print('Live Birth labeled Not Applicable', 
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

#use standard deviations to get low, medium, and high income
income_mean = data_natality['Mean income (dollars)'].mean()
income_stdev = data_natality['Mean income (dollars)'].std()
income_low_threshold = income_mean - income_stdev
income_high_threshold = income_mean + income_stdev
def income_thresholds(i):
    if i < income_low_threshold:
        return 1
    elif i > income_high_threshold:
        return 3
    else:
        return 2
data_natality['Income Category'] = data_natality['Mean income (dollars)'].apply(income_thresholds)


#add BMI categories following:
#Underweight: BMI is less than 18.5
#Normal weight: BMI is 18.5 to 24.9
#Overweight: BMI is 25 to 29.9
#Obese: BMI is 30 or more
def BMI_thresholds(bmi):
    if bmi < 18.5:
        #underweight
        return 1
    elif bmi < 24.9:
        #normal
        return 2
    elif bmi < 29.9:
        #overweight
        return 3
    else:
        #ovese
        return 4
data_natality['BMI Category'] = data_natality['Average Pre-pregnancy BMI'].apply(BMI_thresholds)

    

#%%EDA
#data.info()
describe = data_natality.describe()
numerical_columns = ['Births', 'Average Age of Mother (years)',
                     'Average OE Gestational Age (weeks)', 
                     'Average Birth Weight (grams)', 'Average Pre-pregnancy BMI', 
                     'Average Number of Prenatal Visits',
                     'Average Interval Since Last Live Birth (months)',
                     'Median income (dollars)', 'Mean income (dollars)']
#income is a multimodal distribution
#check for outliers with histograms
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(data_natality[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(data_natality[feature].skew(), 2)}")
plt.tight_layout()
plt.show()
#%%check for outliers with 3 standard deviations away from mean

min_prenatal_visits = data_natality['Average Number of Prenatal Visits'].mean() - (3 * data_natality['Average Number of Prenatal Visits'].std())
max_prenatal_visits = data_natality['Average Number of Prenatal Visits'].mean() + (3 * data_natality['Average Number of Prenatal Visits'].std())
outliers_prenatal_visits = data_natality.loc[(data_natality['Average Number of Prenatal Visits'] > max_prenatal_visits) | (data_natality['Average Number of Prenatal Visits'] < min_prenatal_visits)]
print('prenatal visits outliers:',
      len(outliers_prenatal_visits),
      '{:.1%}'.format(len(outliers_prenatal_visits)/rows))

min_age = data_natality['Average Age of Mother (years)'].mean() - (3 * data_natality['Average Age of Mother (years)'].std())
max_age = data_natality['Average Age of Mother (years)'].mean() + (3 * data_natality['Average Age of Mother (years)'].std())
outliers_age = data_natality.loc[(data_natality['Average Age of Mother (years)'] > max_age) | (data_natality['Average Age of Mother (years)'] < min_age)]
print('age outliers:',
      len(outliers_age),
      '{:.1%}'.format(len(outliers_age)/rows))

min_birth_weight = data_natality['Average Birth Weight (grams)'].mean() - (3 * data_natality['Average Birth Weight (grams)'].std())
max_birth_weight = data_natality['Average Birth Weight (grams)'].mean() + (3 * data_natality['Average Birth Weight (grams)'].std())
outliers_birth_weight = data_natality.loc[(data_natality['Average Birth Weight (grams)'] > max_birth_weight) | (data_natality['Average Birth Weight (grams)'] < min_birth_weight)]
print('birth weight outliers:',
      len(outliers_birth_weight),
      '{:.1%}'.format(len(outliers_birth_weight)/rows))

min_gest_age = data_natality['Average OE Gestational Age (weeks)'].mean() - (3 * data_natality['Average OE Gestational Age (weeks)'].std())
max_gest_age = data_natality['Average OE Gestational Age (weeks)'].mean() + (3 * data_natality['Average OE Gestational Age (weeks)'].std())
outliers_gest_age = data_natality.loc[(data_natality['Average OE Gestational Age (weeks)'] > max_gest_age) | (data_natality['Average OE Gestational Age (weeks)'] < min_gest_age)]
print('gestational age outliers:',
      len(outliers_gest_age),
      '{:.1%}'.format(len(outliers_gest_age)/rows))

min_bmi = data_natality['Average Pre-pregnancy BMI'].mean() - (3 * data_natality['Average Pre-pregnancy BMI'].std())
max_bmi = data_natality['Average Pre-pregnancy BMI'].mean() + (3 * data_natality['Average Pre-pregnancy BMI'].std())
outliers_bmi = data_natality.loc[(data_natality['Average Pre-pregnancy BMI'] > max_bmi) | (data_natality['Average Pre-pregnancy BMI'] < min_bmi)]
print('BMI outliers:',
      len(outliers_bmi),
      '{:.1%}'.format(len(outliers_bmi)/rows))

min_inter = data_natality['Average Interval Since Last Live Birth (months)'].mean() - (3 * data_natality['Average Interval Since Last Live Birth (months)'].std())
max_inter = data_natality['Average Interval Since Last Live Birth (months)'].mean() + (3 * data_natality['Average Interval Since Last Live Birth (months)'].std())
outliers_inter = data_natality.loc[(data_natality['Average Interval Since Last Live Birth (months)'] > max_inter) | (data_natality['Average Interval Since Last Live Birth (months)'] < min_inter)]
print('Interval Since Last Live Birth outliers:',
      len(outliers_inter),
      '{:.1%}'.format(len(outliers_inter)/rows))


min_income = data_natality['Mean income (dollars)'].mean() - (3 * data_natality['Mean income (dollars)'].std())
max_income = data_natality['Mean income (dollars)'].mean() + (3 * data_natality['Mean income (dollars)'].std())
outliers_income = data_natality.loc[(data_natality['Mean income (dollars)'] > max_income) | (data_natality['Mean income (dollars)'] < min_income)]
print('Income outliers:',
      len(outliers_income),
      '{:.1%}'.format(len(outliers_income)/rows))


correlation_matrix = data_natality[['Number of Prenatal Visits Code', 
                                    'NICU Admission Code', 'Births', 
                                    'Average Age of Mother (years)', 
                                    'Average OE Gestational Age (weeks)', 
                                    'Average Birth Weight (grams)', 
                                    'Average Pre-pregnancy BMI', 
                                    'Average Number of Prenatal Visits', 
                                    'Average Interval Since Last Live Birth (months)', 
                                    'Median income (dollars)', 'Mean income (dollars)', 
                                    'Age of Mother encoded', 'Birth Weight Category', 
                                    'Interval Category', 'Income Category', 
                                    'BMI Category']].corr()
#features with higher correlations:
#gestational age NICU admission
#birth weight (and category) and NICU admission
#avg age of mother and avg interval since last birth
#birth weight (and cat) and gestational age


#cluster the data to look for relationships
#PCA then k-means
#rand stat
