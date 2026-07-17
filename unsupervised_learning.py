import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Prepare the data
experience = np.array([[1], [2], [3], [5], [7], [10]])   # Years of experience
salary = np.array([30, 35, 50, 80, 95, 105])            # Salary (in $1000s)

# Step 2: Transform input into polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
experience_poly = poly.fit_transform(experience)

# Step 3: Create and train the model
model = LinearRegression()
model.fit(experience_poly, salary)

# Step 4: Generate points for a smooth curve line
# We create 100 points between 1 and 10 so the line looks perfectly smooth, not jagged
X_smooth = np.linspace(1, 10, 100).reshape(-1, 1)
X_smooth_poly = poly.transform(X_smooth)
y_smooth_pred = model.predict(X_smooth_poly)

# Step 5: Plotting the Scatter Plot and the Curve
plt.figure(figsize=(8, 5))

# 1. Plot the actual data points as a scatter plot
plt.scatter(experience, salary, color='red', label='Actual Data (Scatter)', s=100)

# 2. Plot the Polynomial Regression curve
plt.plot(X_smooth, y_smooth_pred, color='blue', linewidth=2, label='Polynomial Regression Curve (Degree 2)')

# Customize the plot appearance
plt.title('Salary vs Experience (Polynomial Regression)', fontsize=14)
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary (in $1000s)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()