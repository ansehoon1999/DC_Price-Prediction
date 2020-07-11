from pandas import DataFrame
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression as lin
from sklearn.model_selection import cross_val_score
df = pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_after_preprocessing.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['grade'], axis=1, inplace=True)

#use minmaxscaler to normalize data
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
price_list = covMatrix[12].values.tolist()
cov = pd.DataFrame({"value":price_list},index =
                   ['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','price'])
cov = cov.sort_values(by=['value'], axis=0, ascending=False)

print(cov)
#delete less relevant columns and covmatrix value of itself(price*price)
#delete columns that has abs(value) < 0.1
indexes = cov[abs(cov["value"]) < 0.001 ].index
cov = cov.drop(indexes)

indexes = cov[cov["value"].index == "price" ].index
cov = cov.drop(indexes)



#record column names
scaled_df.columns =['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','price']
y=np.array(pd.DataFrame(df,columns=['price']))


#x = scaled_df's relevant columns(except price and abs(value)<0.1)
x=np.array(pd.DataFrame(scaled_df,columns=cov.index.tolist()))
#y = df's price(which has not scaled)
y=np.array(pd.DataFrame(df,columns=['price']))

#linear regression model training
reg = lin(fit_intercept=True, normalize=False, n_jobs=None)
reg.fit(x, y)

#print accuracy of linear regression model
accuracy = cross_val_score(reg, x, y, cv = 5)
acc = round(sum(accuracy)/5,2)
print ("Linear Regression test file accuracy:"+str(acc))

#compare predicted price and real price
y_pred = reg.predict(x)
plt.scatter(y,y_pred)
#plot x = y 
plt.plot([0,5000000],[0,5000000],color='black',lw=2,linestyle='solid')
plt.xlabel("real price($100,000)")
plt.ylabel("predicted price($100,000)")
plt.show()
