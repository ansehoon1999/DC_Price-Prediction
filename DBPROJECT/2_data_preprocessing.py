import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt 
#read data from csv file
data=pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_data_original.csv")
data_copy=data
#na값 채우기

data_copy['waterfront'].fillna(axis=0,method='ffill',inplace=True)
#fill data in waterfront feature NA data using ffill method
data_copy['district'].fillna(axis=0,method='ffill',inplace=True)
#fill data in district feature NA data using ffill method
median=data_copy['bedrooms'].median()
data_copy['bedrooms'].fillna(median,inplace=True)
#fill data in district feature NA data using bedroom's median cost
median=data_copy['bathrooms'].median()
data_copy['bathrooms'].fillna(median,inplace=True)
#fill data in district feature NA data using bedroom's median cost

median=data_copy['sqft_above'].median()
data_copy['sqft_above'].fillna(median,inplace=True)
#fill data in district feature NA data using sqft_above median cost

median=data_copy['sqft_living'].median()
data_copy['sqft_living'].fillna(median,inplace=True)
#fill data in district feature NA data using sqft_living median cost

median=data_copy['sqft_lot'].median()
data_copy['sqft_lot'].fillna(median,inplace=True)
#fill data in district feature NA data using sqft_lot median cost

average=data_copy['price'].mean().round()
data_copy['price'].fillna(average,inplace=True)
#fill data in district feature NA data using sqft_lot median cost

water=[]
dist=[]
condition=[]
grade=[]


#change category feature to numeric feature
for i in range(len(data_copy)):
    if data_copy.at[i,'waterfront']=='NO': #change 'NO' to 0 
        water.append(0)
    elif data_copy.at[i,'waterfront']=='YES':#change 'YES' to 1 
        water.append(1)

    if data_copy.at[i,'district']=='suburb':#change 'suburb' to 0 
        dist.append(0)
    elif data_copy.at[i,'district']=='normal':#change 'district' to 1 
        dist.append(1)
    elif data_copy.at[i,'district']=='subcity':#change 'subcity' to 2 
        dist.append(2)
    elif data_copy.at[i,'district']=='city':#change 'city' to 3 
        dist.append(3)
    elif data_copy.at[i,'district']=='mainStreet':#change 'mainStreet' to 4 
        dist.append(4)

    if data_copy.at[i,'condition']=='poor':#change 'poor' to 0 
        condition.append(0)
    elif data_copy.at[i,'condition']=='notgood':#change 'not good' to 1
        condition.append(1)
    elif data_copy.at[i,'condition']=='normal':#change 'normal' to 2 
        condition.append(2)
    elif data_copy.at[i,'condition']=='good':#change 'good' to 3 
        condition.append(3)
    elif data_copy.at[i,'condition']=='excellent':#change 'excellent' to 4 
        condition.append(4)

    if data_copy.at[i,'grade']=='low':#change 'low' to 0 
        grade.append(0)
    elif data_copy.at[i,'grade']=='normal':#change 'normal' to 1 
        grade.append(1)
    elif data_copy.at[i,'grade']=='high':#change 'high' to 2 
        grade.append(2)
    elif data_copy.at[i,'grade']=='luxury':#change 'luxury' to 3 
        grade.append(3)

#store changed data to data_copy
data_copy['waterfront']=water
data_copy['district']=dist
data_copy['condition']=condition
data_copy['grade']=grade

#discard the data that we will never need in this service
data_copy.drop(['long'],axis=1,inplace=True) #discard long feature data
data_copy.drop(['lat'],axis=1,inplace=True) #discard lat feature data
data_copy.drop(['date'],axis=1,inplace=True) #discard date feature data

#because we will use sqft_living (sqft_above + sqft_basement)
#so we don't need sqft_above, sqft_basement
data_copy.drop(['sqft_above'],axis=1,inplace=True) 
data_copy.drop(['sqft_basement'],axis=1,inplace=True)


data_copy.to_csv("C:\\Users\\PC\\Desktop\\\\DBPROJECT\\kc_house_after_preprocessing.csv")
