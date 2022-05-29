# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:11:01 2022

@author: archi
"""

#importing various libraries that will be used in this code
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import altair as alt

 
#reading the data from the given csv file and storing it 
df = pd.read_csv("cars_engage_2022.csv")
print(df.head())


#heading of our webpage
st.title("Welcome to my Carsworld!!!")
st.subheader("This is an interactive webpage where u can see various conclusions regarding the cars dataset of 2022.")
st.subheader("                       ")
st.subheader("                       ")

#cleaning the raw data 
df['car'] = df.Make + ' ' + df.Model
features = ['Make','Model','car','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
     'ARAI_Certified_Mileage','Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']
df_full = df.copy()
df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace('Rs. ','',regex=False)
df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace(',','',regex=False)
df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].astype(int)
df = df[features]
df = df[~df.ARAI_Certified_Mileage.isnull()]
df = df[~df.Make.isnull()]
df = df[~df.Width.isnull()]
df = df[~df.Cylinders.isnull()]
df = df[~df.Wheelbase.isnull()]
df = df[~df['Fuel_Tank_Capacity'].isnull()]
df = df[~df['Seating_Capacity'].isnull()]
df = df[~df['Torque'].isnull()]
df['Height'] = df['Height'].str.replace(' mm','',regex=False).astype(float)
df['Length'] = df['Length'].str.replace(' mm','',regex=False).astype(float)
df['Width'] = df['Width'].str.replace(' mm','',regex=False).astype(float)
df['Wheelbase'] = df['Wheelbase'].str.replace(' mm','',regex=False).astype(float)
df['Fuel_Tank_Capacity'] = df['Fuel_Tank_Capacity'].str.replace(' litres','',regex=False).astype(float)
df['Displacement'] = df['Displacement'].str.replace(' cc','',regex=False)
df.loc[df.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
df.loc[df.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
df['ARAI_Certified_Mileage'] = df['ARAI_Certified_Mileage'].str.replace(' km/litre','',regex=False).astype(float)
df.Number_of_Airbags.fillna(0,inplace= True)
df['price'] = df['Ex-Showroom_Price'] * 0.014
df.drop(columns='Ex-Showroom_Price', inplace= True)
df.price = df.price.astype(int)
HP = df.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
HP = HP.apply(lambda x: round(x,2))
TQ = df.Torque.str.extract(r'(\d{1,4}).*').astype(int)
TQ = TQ.apply(lambda x: round(x,2))
df.Torque = TQ
df.Power = HP
df.Doors = df.Doors.astype(int)
df.Seating_Capacity = df.Seating_Capacity.astype(int)
df.Number_of_Airbags = df.Number_of_Airbags.astype(int)
df.Displacement = df.Displacement.astype(int)
df.Cylinders = df.Cylinders.astype(int)
df.columns = ['make', 'model','car', 'variant', 'body_type', 'fuel_type', 'fuel_system','type', 'drivetrain', 'displacement', 'cylinders',
              'mileage', 'power', 'torque', 'fuel_tank','height', 'length', 'width', 'doors', 'seats', 'wheelbase','airbags', 'price']


#the following chart will be used to find out the top 3 most popular types of cars that people prefer to use
st.subheader("1. Types of cars")
df['freq'] = df.groupby('body_type')['body_type'].transform('count')
chart = (
    alt.Chart(df).mark_bar().encode(x=alt.X("freq", type="quantitative",title=""),
                                    y=alt.Y("body_type",type="nominal",title=""),
                                    color=alt.Color("freq", type="quantitative",title=""),
                                    ))
st.altair_chart(chart)
st.subheader("Conclusion")
st.markdown("From the above graph, we conclude that **SUVs** are the most popular type of Cars, followed by **Hatchbacks** in the 2nd Place and **Sedans** in the 3rd.")
st.markdown("     ")
st.subheader("    ")


#the following chart depicts the fuel which is used by various cars
st.subheader("2. Car count based on the fuel used")
fir = plt.figure(figsize=(15,8))
sns.countplot(data=df, x='fuel_type')
plt.title('Cars count by engine fuel type',fontsize=18)
plt.xlabel('Fuel Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cars Count', fontsize=16);
st.plotly_chart(fir, use_container_width=True)
st.markdown("0 - Petrol, 1 - Diesel, 2 - CNG+Petrol, 3 - Hybrid")
st.markdown(" ")
st.subheader("Conclusion")
st.markdown("From the above graph, we conclude that **petrol** based and **diesel** based vehicles are still the most abundant in the current scenario.")
st.markdown("     ")
st.subheader("    ")

#the following chart shows us the distribution of cars on the basis of their engine size
st.subheader("3. Distribution of the cars on the basis on their engine size")
plot3 = plt.figure(figsize=(15,8))
sns.histplot(data=df, x='displacement',bins=10)
plt.title('Cars by engine size (in CC)',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Engine Size")
st.plotly_chart(plot3, use_container_width=True)
st.subheader("Conclusion")
st.markdown("From the above graph, we conclude that most of the engine size is between the range of **500cc** to **30000cc**.")
st.markdown("     ")
st.subheader("    ")


#the following chart shows us the distribution of cars based on their horsepowers
st.subheader("4. Bar chart representing the horsepower of cars")
plot4=plt.figure(figsize=(15,8))
sns.histplot(data=df, x='power')
plt.title('Horsepower of Cars',fontsize=18);
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
st.plotly_chart(plot4, use_container_width=True)
st.subheader("Conclusion")
st.markdown("From the above graph, we conclude that most of the engine size is between the range of **60HP** to **180HP**.")
st.markdown("     ")
st.subheader("    ")


#the following scatterplot represents the relation between the power and price of a vehicle
st.subheader("5. Scatterplot representing the relation between the power and price of a car")
plot5 = plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x='power', y='price');
plt.xticks(fontsize=13);
plt.yticks(fontsize=13)
plt.xlabel('power',fontsize=15)
plt.ylabel('price',fontsize=15)
plt.title('Relationship between Power and Price',fontsize=20);
st.plotly_chart(plot5, use_container_width=True)
st.subheader("Conclusion")
st.markdown("It is pretty evident from our plot that as the power of the vehicle increases the price of the it also increases")
st.markdown("     ")
st.subheader("    ")

#using the linear regression model to calculate the required co efficients
st.subheader("6. Linear regression model to predict the coefficients of the linear equation between price and power")
x = df.power
y = df.price
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
pred = model.predict(x)
st.write("Model Coefficient: " + str(model.coef_))
st.write("Model Intercept: " + str(model.intercept_))
plot6 = plt.figure(figsize=(15,8))
plt.scatter(x,y,s=5, label='training')
plt.scatter(x,pred,s=5, label='prediction')
plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.legend()
plt.show()
st.plotly_chart(plot6, use_container_width=True)
st.markdown("Here the **orange** dotted line represents the **prediction model** whereas the **training** is represented by the **blue dotted** scatter plot")
st.markdown("     ")
st.subheader("    ")


#here we will be using a polynomial model to estimate the relationship between power and price
st.subheader("7. Polynomial model to predict the relationship between price and power")
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x)
model.fit(poly_features,y)
new_pred = model.predict(poly_features)
plot7 = plt.figure(figsize=(15,8))
plt.scatter(x,y,s=5, label='training')
plt.scatter(x,new_pred,c='red', label='prediction')
plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.legend()
plt.show()
st.plotly_chart(plot7, use_container_width=True)
st.markdown("Here the **red** curve represents the **polynominal prediction** and the **blue dotted plot** represents the training")
st.markdown("     ")
st.subheader("    ")


#this scatter plot will represent the relationship between mileage and price
st.subheader("8. Scatter plot to represent the relationship between mileage and price of a vehicle")
plot8=plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x='mileage', y='price');
st.plotly_chart(plot8, use_container_width=True)
st.subheader("Conclusion")
st.markdown("From the above graph, we conclude that as price increases mileage decreases.")
st.markdown("     ")
st.subheader("    ")



#creating an interactive chart using the selectbox 
st.subheader("9. Interactive Chart")
st.markdown("In this barchart you can select your required criteria in the checkbox provided and the appropriate chart will be shown")
choice = st.sidebar.selectbox("Desired Criteria", df.columns)
fir2 = plt.figure(figsize=(10,10))
sns.countplot(data=df, y=df[choice])
plt.title('Type of Car',fontsize=30)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
st.plotly_chart(fir2, use_container_width = True)