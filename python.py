from sklearn.tree import DecisionTreeClassifier
X = [[150, 7], [170, 6], [140, 8], [200, 4], [210, 3], [190, 5]]
y = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange"]
model=DecisionTreeClassifier()
model.fit(X,y)
print(model.predict([[190,6]]))
