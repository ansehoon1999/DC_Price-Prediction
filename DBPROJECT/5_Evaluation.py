
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import BaggingClassifier

data=pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_after_preprocessing.csv")
train=data[['sqft_living','district','bathrooms','bedrooms','floors','yr_built','grade']]
test=data[['sqft_living','district','bathrooms','bedrooms','floors','yr_built','grade']]

X=train[['sqft_living','district','bathrooms','bedrooms','floors','yr_built']]
y=train['grade']

X_test=X

X=np.array(X)
y=np.array(y)

X_test=np.array(X_test)

model=BaggingClassifier()
model.fit(X,y)

result=model.predict(X_test)

print(result)

y_true=data['grade']

y_true=np.array(y_true)
y_true=y_true.tolist()

data_map={'y_predict':result, 'y_actual':y_true}
df=pd.DataFrame(data_map,columns=['y_actual','y_predict'])
confusion_matrix=pd.crosstab(df['y_actual'],df['y_predict'],rownames=['Actual'],
                             colnames=['Predicted'])
sn.heatmap(confusion_matrix,annot=True)
plt.show()
