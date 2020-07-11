from pandas import DataFrame
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_after_preprocessing.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['price'], axis=1, inplace=True)

#use standardscaler to normalize data
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df)
#making covmatrix
covMatrix = pd.DataFrame.cov(scaled_df)
top_cov_features = covMatrix.index
plt.figure(figsize=(20,20))
g = sns.heatmap(covMatrix[top_cov_features], annot=True, cmap="RdYlGn")
plt.show()
#covMatrix's 12th column has price's relation score
#make a new dataframe to store value(relation score) and sort in descending order
value =pd.Series(covMatrix[12])
grade_list = covMatrix[12].values.tolist()
cov = pd.DataFrame({"value":grade_list},index =
                   ['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','grade'])
cov = cov.sort_values(by=['value'], axis=0, ascending=False)

print(cov)
#delete less relevant columns and covmatrix value of itself(price*price)
#delete columns that has abs(value) < 0.015
indexes = cov[abs(cov["value"]) < 0.015 ].index
cov = cov.drop(indexes)
print(cov)
indexes = cov[cov["value"].index == "grade"].index
cov = cov.drop(indexes)

#record column names
scaled_df.columns =['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','grade']

#knn algorithm
knn = KNeighborsClassifier()
#x = scaled_df's relevant columns(except price and abs(value)<0.015)
x=np.array(pd.DataFrame(scaled_df,columns=cov.index.tolist()))
#y = df's price(which has not scaled) change y_r -> y
y_r= np.array(pd.DataFrame(df,columns=['grade']))
y = []
for i in range(len(y_r)) :
    y.append(y_r[i][0])
#knn algorithm and GridSearchCV
param_grid = {'n_neighbors': np.arange(1,25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(x, y)
#y_pred = knn_gscv.predict(x)

#print accuracy of KNN algorithm
accuracy = cross_val_score(knn, x, y, cv = 5)
acc = round(sum(accuracy)/5, 2)
print ("KNN test file accuracy:"+str(acc))
