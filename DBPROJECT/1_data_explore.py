import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#bedrooms,bathrooms,sqft_living,waterfront(pie),district(pie),price,floor
data=pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_data_original.csv")

#grade
grade=data['grade']
grade_array=np.array(grade)
normal=0
low=0
high=0
luxury=0
for i in range(len(grade_array)):
    if grade_array[i]=='low':
        low+=1
    elif grade_array[i]=='normal':
        normal+=1
    elif grade_array[i]=='high':
        high+=1
    elif grade_array[i]=='luxury':
        luxury+=1
level=['low','normal','high','luxury']
grade=[low,normal,high,luxury]
plt.pie(grade,labels=level,autopct='%1.1f%%')
plt.title('Grade')
plt.show()


#bedrooms
bed=data['bedrooms']
bed_array=np.array(bed)
plt.hist(bed_array,bins=[0,1,2,3,4,5,6,7])
plt.xlabel('bedroom')
plt.title('Bedroom')
plt.show()

#bathrooms
bath=data['bathrooms']
bath_array=np.array(bath)
plt.hist(bath_array,bins=[0,1,2,3,4,5,6,7,8])
plt.xlabel('bathroom')
plt.title('Bathroom')
plt.show()

#sqft_living
space=data['sqft_living']
space_array=np.array(space)
plt.hist(space_array,bins=16)
plt.xlabel('Square foot')
plt.title('Space')
plt.show()

#waterfront
water=data['waterfront']
water_array=np.array(water)
yes=0
no=0
for i in range(len(water_array)):
    if water_array[i]=='YES':
        yes+=1
    elif water_array[i]=='NO':
        no+=1
level=['YES','NO']
waterfront=[yes,no]
plt.pie(waterfront,labels=level,autopct='%1.1f%%')
plt.title('WaterFront')
plt.show()



#District
dist=data['district']
dist_array=np.array(dist)
subu=0
subc=0
nor=0
ci=0
main=0
for i in range(len(dist_array)):
    if dist_array[i]=='suburb':
        subu+=1
    elif dist_array[i]=='subcity':
        subc+=1
    elif dist_array[i]=='normal':
        nor+=1
    elif dist_array[i]=='city':
        ci+=1
    elif dist_array[i]=='mainStreet':
        main+=1
        
level=['suburb','subcity','normal','city','mainStreet']
district=[subu,subc,nor,ci,main]
plt.pie(district,labels=level,autopct='%1.1f%%')
plt.title('District')
plt.show()

#price
price=data['price']
price_array=np.array(price)
plt.hist(price_array,bins=8)
plt.xlabel('price($)')
plt.title('Price')
plt.show()

#floor
floor=data['floors']
floor_array=np.array(floor)
plt.hist(floor_array,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4])
plt.xlabel('floor')
plt.title('Floor')
plt.show()
