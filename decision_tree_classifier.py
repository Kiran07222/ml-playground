from sklearn.tree import DecisionTreeClassifier
import pandas as pd
data = {
    "Study_Hours": [2, 3, 4, 5, 6],
    "Attendance": [60, 70, 80, 85, 90],
    "Pass": [0, 0, 1, 1, 1]
}
df=pd.DataFrame(data)
X = df[["Study_Hours", "Attendance"]]  # Features
y = df["Pass"]                         # Label

model=DecisionTreeClassifier()
model.fit(X,y)
print(model.predict([[4.5,80]]))
