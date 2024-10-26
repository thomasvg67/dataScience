import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
student = pd.read_csv('student_scores.csv')

# Explore the data
print(student.head())
print(student.describe())
print(student.info())

# Visualize the data
Xax = student.iloc[:, 0]
Yax = student.iloc[:, 1]
plt.scatter(Xax, Yax)
plt.xlabel("No. of hours")
plt.ylabel("Score")
plt.title("Student scores")
plt.show()

# Prepare features and target variable
x = student.iloc[:, :-1]
y = student.iloc[:, 1]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create and train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Print model coefficients
print('INTERCEPT = ', regressor.intercept_)
print('COEFFICIENT = ', regressor.coef_)

# Make predictions
y_pred = regressor.predict(x_test)

# Calculate and print metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Print actual vs predicted values
for (i, j) in zip(y_test, y_pred):
    if i != j:
        print("Actual value:", i, "Predicted value:", j)

print("Number of mislabeled points from test data set:", (y_test != y_pred).sum())
