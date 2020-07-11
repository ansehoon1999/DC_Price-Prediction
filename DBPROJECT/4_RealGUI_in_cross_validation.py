import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression as lin

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame

from tkinter import * 
from PIL import ImageTk, Image
bgcolor = 'floral white'
win = Tk()
win.title('predict house price in Seatlle')
win.geometry('950x400+0+0')
win.configure(bg=bgcolor)

def predict():

    #value change to show your result
    if district.get() == 'suburb' :
        district_get = 0
    elif district.get() == 'normal' :
        district_get = 1
    elif district.get() == 'subcity' :
        district_get = 2
    elif district.get() == 'city' :
        district_get = 3
    elif district.get() == 'mainStreet' :
        district_get = 4
    #value change to show your result
    if waterfront.get() == 'yes' :
        waterfront_get = 1
    elif waterfront.get() == 'no' :
        waterfront_get = 0

    #process to change shape to predict the grade
    scaler = preprocessing.StandardScaler()
    dft2 = df1
    dft2.loc[len(dft2)] =[np.nan, bedroom.get(), bathroom.get(), sqft.get(),
                        np.nan, floor.get(), district_get, np.nan, yr_built.get(), np.nan, np.nan, np.nan, np.nan]
    
    scaled_dft2 = scaler.fit_transform(dft2)
    temp1 = scaled_dft2[len(dft2)-1][:]
    pred1 = []
    pred1 = [temp1[3], temp1[2], temp1[5],
        temp1[8], temp1[1], temp1[6]] #grade

        
    #process to change shape to predict the price
    dft = df1
    dft.loc[len(dft)] =[np.nan, bedroom.get(), bathroom.get(), sqft.get(),
                        np.nan, floor.get(), district_get, waterfront_get, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    scaled_dft = scaler.fit_transform(dft)
    temp1 = scaled_dft[len(dft)-1][:]
    pred2 = []
    pred2 = [temp1[3], temp1[2], temp1[7],
             temp1[1], temp1[5], temp1[6]] #price
    
    print('pred1', pred1)
    print('pred2', pred2)
    pg = knn_gscv.predict(np.array([pred1])) #grade
    pp = reg.predict([pred2]) #price
    dft.dropna(inplace=True)
    dft2.dropna(inplace=True)

    #predicted grade is pg (number)
    print(pg)
    if pg[0] == 0 :
        pgs = 'low'
    elif pg[0] == 1 :
        pgs = 'normal'
    elif pg[0] == 2 :
        pgs = 'high'
    elif pg[0] == 3 :
        pgs = 'luxury'
    print(pgs)
    ggrade.set(pgs)
    g_accuracy.set(acc1)
    p_accuracy.set(acc2)
    pprice.set(pp[0][0].round())
    
def exite():
    win.destroy()
def reset():
    sqft.set('')
    bathroom.set('')
    bedroom.set('')
    floor.set('')
    yr_built.set('')
    district.set('')
    waterfront.set('')
    
#====================================================================


#=====================Standard Scaler/Multiple Regression===============

df1 = pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_after_preprocessing.csv")
df1.drop(['Unnamed: 0'], axis=1, inplace=True)
df1.drop(['grade'], axis=1, inplace=True)
#use standardscaler to normalize data
scaler1 = preprocessing.StandardScaler()
scaled_df1 = scaler1.fit_transform(df1)
scaled_df1 = pd.DataFrame(scaled_df1)
covMatrix1 = pd.DataFrame.cov(scaled_df1)

#covMatrix's 12th column has price's relation score
#make a new dataframe to store value(relation score) and sort in descending order
value1 =pd.Series(covMatrix1[12])
price_list = covMatrix1[12].values.tolist()
cov1 = pd.DataFrame({"value":price_list},index =
                   ['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','price'])
cov1 = cov1.sort_values(by=['value'], axis=0, ascending=False)

#delete less relevant columns and covmatrix value of itself(price*price)
#delete columns that has abs(value) < 0.1
indexes1 = cov1[abs(cov1["value"]) < 0.2 ].index
cov1 = cov1.drop(indexes1)

indexes1 = cov1[cov1["value"].index == "price"].index
cov1 = cov1.drop(indexes1)

#record column names
scaled_df1.columns =['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','price']

#x = scaled_df's relevant columns(except price and abs(value)<0.1)
x1=np.array(pd.DataFrame(scaled_df1,columns=cov1.index.tolist()))
#y = df's price(which has not scaled)
y1=np.array(pd.DataFrame(df1,columns=['price']))
#linear regression model valdiation
reg = lin(fit_intercept=True, normalize=False, n_jobs=None)
reg.fit(x1, y1)

#print accuracy of linear regression model
accuracy1 = cross_val_score(reg, x1, y1, cv = 5)
acc1 = round(sum(accuracy1)/5, 2)



#=====================Standard Scaler/KNN algorithm===============
df2 = pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_after_preprocessing.csv")
df2.drop(['Unnamed: 0'], axis=1, inplace=True)
df2.drop(['price'], axis=1, inplace=True)

#use standardscaler to normalize data
scaler2 = preprocessing.StandardScaler()
scaled_df2 = scaler2.fit_transform(df2)
scaled_df2 = pd.DataFrame(scaled_df2)
#making covmatrix
covMatrix2 = pd.DataFrame.cov(scaled_df2)
top_cov_features2 = covMatrix2.index

#covMatrix's 12th column has price's relation score
#make a new dataframe to store value(relation score) and sort in descending order
value2 =pd.Series(covMatrix2[12])
grade_list = covMatrix2[12].values.tolist()
cov2 = pd.DataFrame({"value":grade_list},index =
                   ['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','grade'])
cov2 = cov2.sort_values(by=['value'], axis=0, ascending=False)

print(cov2)
#delete less relevant columns and covmatrix value of itself(price*price)
#delete columns that has abs(value) < 0.2
indexes2 = cov2[abs(cov2["value"]) < 0.2 ].index
cov2 = cov2.drop(indexes2)
print(cov2)
indexes2 = cov2[cov2["value"].index == "grade"].index
cov2 = cov2.drop(indexes2)

#record column names
scaled_df2.columns =['id','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront'
                    ,'district','condition','yr_built','yr_renovated','zipcode','grade']

#knn algorithm
knn = KNeighborsClassifier()
#x = scaled_df's relevant columns(except price and abs(value)<0.2)
x2=np.array(pd.DataFrame(scaled_df2,columns=cov2.index.tolist()))
#y = df's price(which has not scaled) change y_r -> y
y_r= np.array(pd.DataFrame(df2,columns=['grade']))
y2 = []
for i in range(len(y_r)) :
    y2.append(y_r[i][0])
#knn algorithm and GridSearchCV
param_grid = {'n_neighbors': np.arange(1,25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(x2, y2)
#y_pred = knn_gscv.predict(x)

#print accuracy of KNN algorithm
accuracy2 = cross_val_score(knn, x2, y2, cv = 5)
acc2 = round(sum(accuracy2)/5, 2)

#================================================================

tops = Frame(win, width = 1000, height=50, bg=bgcolor, relief=SUNKEN)

tops.pack(side =TOP)

f1 = Frame(win, width=600, height = 700, bg=bgcolor, relief=SUNKEN)

f1.pack(side=LEFT)
n= Frame(win, width=30, height = 700,bg=bgcolor,relief=SUNKEN)
n.pack(side=LEFT)
f3 = Frame(win, width=100, height = 700,bg=bgcolor,relief=SUNKEN)
f3.pack(side=LEFT)
imgLabel = Label(f3,bg=bgcolor)
img = PhotoImage(file="C:\\Users\\PC\\Desktop\\next.png")
imgLabel.config(image=img)
imgLabel.pack()

f2 = Frame(win, width = 350, height=700, bg=bgcolor,relief=SUNKEN)
f2.pack(side=LEFT)

lbl = Label(tops, font=('arial', 20, 'bold'), bg=bgcolor, text='New Built House in Seattle', fg='Steel Blue', bd=10, anchor='w')
lbl.grid(row=0, column=0)
#===================================================column 0, 1
sqft = IntVar()
bathroom = IntVar()
bedroom = IntVar()
floor = IntVar()
yr_built = IntVar()
district = StringVar()
waterfront = StringVar()

lbl1 = Label(f1, font=('arial', 10, 'bold'), bg=bgcolor, text='house sqft', bd=16, anchor='w')
lbl1.grid(row=0, column=0)
entry1 = Entry(f1, font=('arial', 10, 'bold'), textvariable=sqft, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry1.grid(row=0, column=1)


lbl2 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor, text='bathroom', bd=16, anchor='w')
lbl2.grid(row=1, column=0)
entry2 = Entry(f1, font=('arial', 10, 'bold'), textvariable=bathroom, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry2.grid(row=1, column=1)


lbl3 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor,text='bedroom', bd=16, anchor='w')
lbl3.grid(row=2, column=0)
entry3 = Entry(f1, font=('arial', 10, 'bold'), textvariable=bedroom, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry3.grid(row=2, column=1)


lbl4 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor,text='floor', bd=16, anchor='w')
lbl4.grid(row=3, column=0)
entry4 = Entry(f1, font=('arial', 10, 'bold'), textvariable=floor, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry4.grid(row=3, column=1)
#===================================================column 2, 3


lbl5 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor,text='built year', bd=16, anchor='w')
lbl5.grid(row=0, column=2)
entry5 = Entry(f1, font=('arial', 10, 'bold'), textvariable=yr_built, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry5.grid(row=0, column=3)


lbl6 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor,text='district', bd=16, anchor='w')
lbl6.grid(row=1, column=2)
entry6 = Entry(f1, font=('arial', 10, 'bold'), textvariable=district, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry6.grid(row=1, column=3)


lbl7 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor,text='waterfront', bd=16, anchor='w')
lbl7.grid(row=2, column=2)
entry7 = Entry(f1, font=('arial', 10, 'bold'), textvariable=waterfront, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry7.grid(row=2, column=3)


where = StringVar()
where.set('Seattle')
lbl8 = Label(f1, font=('arial', 10, 'bold'),  bg=bgcolor,text='where', bd=16, anchor='w')
lbl8.grid(row=3, column=2)
entry8 = Entry(f1, font=('arial', 10, 'bold'), textvariable=where, bd=10, insertwidth=4, bg='powder blue', justify = 'right')
entry8.grid(row=3, column=3)
#===================================================button=============================

btn1 = Button(f1, padx=16, pady=8, bd=10, fg='black',font=('arial', 10, 'bold'), width=6,
              text='predict', bg='powder blue', command=predict).grid(row=7,column=1)
btn2 = Button(f1, padx=16, pady=8, bd=10, fg='black',font=('arial', 10, 'bold'), width=6,
              text='reset', bg='powder blue', command=reset).grid(row=7,column=2)
btn3 = Button(f1, padx=16, pady=8, bd=10, fg='black',font=('arial', 10, 'bold'), width=6,
              text='exit', bg='powder blue', command=exite).grid(row=7,column=3)
#=======================================================================================
pprice = IntVar()
lbl9 = Label(f2, font=('arial', 10, 'bold'), bg=bgcolor, text='price', bd=16, anchor='w')
lbl9.grid(row=0, column=0)
entry9 = Entry(f2, font=('arial', 10, 'bold'), textvariable=pprice, bd=10, insertwidth=4, bg='white', justify = 'right')
entry9.grid(row=0, column=1)


ggrade = StringVar()
lbl10 = Label(f2, font=('arial', 10, 'bold'), bg=bgcolor, text='grade', bd=16, anchor='w')
lbl10.grid(row=1, column=0)
entry10 = Entry(f2, font=('arial', 10, 'bold'), textvariable=ggrade, bd=10, insertwidth=4, bg='white', justify = 'right')
entry10.grid(row=1, column=1)

p_accuracy = DoubleVar()
lbl11 = Label(f2, font=('arial', 10, 'bold'), bg=bgcolor, text='grade accuracy', bd=16, anchor='w')
lbl11.grid(row=2, column=0)
entry11 = Entry(f2, font=('arial', 10, 'bold'), textvariable=p_accuracy, bd=10, insertwidth=4, bg='white', justify = 'right')
entry11.grid(row=2, column=1)

g_accuracy = DoubleVar()
lbl12 = Label(f2, font=('arial', 10, 'bold'),bg=bgcolor,  text='high accuracy', bd=16, anchor='w')
lbl12.grid(row=3, column=0)
entry12 = Entry(f2, font=('arial', 10, 'bold'), textvariable=g_accuracy, bd=10, insertwidth=4, bg='white', justify = 'right')
entry12.grid(row=3, column=1)

win.mainloop()


