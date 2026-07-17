from sklearn.linear_model import LinearRegression
hours = [[1], [2], [3], [4]]       # input feature: Study Hours
scores = [40, 50, 65, 70]
model=LinearRegression()
model.fit(hours,scores)
print("enter the data: ")
x=int(input())
print(model.predict([[x]]))
