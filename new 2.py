
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
model=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=1)
model.fit(X_train,y_train)
pre1=model.predict(X_test)
print(r2_score(pre1,y_test))
