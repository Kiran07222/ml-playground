from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


df=pd.read_csv("C:/Users/nani3/Desktop/indian_bikes_dataset_1000.csv")
X=df[['cc','top_speed_kmh','mileage_kmpl']]
y=df['on_road_price_inr']

X_train,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)
model=LinearRegression()
model2=DecisionTreeRegressor()
model3 =RandomForestRegressor()
print(model.fit(X_train,y_train))
print(model2.fit(X_train,y_train))
print(model3.fit(X_train,y_train))
prediction2=model2.predict(X_test)
prediction3=model3.predict(X_test)
prediction=model.predict(X_test)
print("linearregression: " , r2_score(y_test,prediction))
print("linearregression: " , mean_squared_error(y_test,prediction))
print("linearregression: " , mean_absolute_error(y_test,prediction))
print("Decisiontree  : " , r2_score(y_test,prediction2))
print("Decisiontree: " , mean_squared_error(y_test,prediction2))
print("Decisiontree: " , mean_absolute_error(y_test,prediction2))
print("Random forest  : " , r2_score(y_test,prediction3))
print("Random forest : " , mean_squared_error(y_test,prediction3))
print("Random forest : " , mean_absolute_error(y_test,prediction3))
