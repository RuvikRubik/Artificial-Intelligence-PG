import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)
def MSE(x, y, theta):
    error = 0
    size = len(x)
    for i in range(size):
        error += (float(theta[0]) + float(theta[1]) * x[i] - y[i]) ** 2
    error /= size
    return error
# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]
X_b = np.c_[np.ones((len(x_train), 1)), x_train]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

# TODO: calculate error
print("training: "+str(MSE(x_train, y_train, theta_best)))
print("test: "+str(MSE(x_test, y_test, theta_best)))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()


# TODO: standarization
x_train_normalized= (x_train - np.mean(x_train)) / np.std(x_train)
y_train_normalized = (y_train - np.mean(y_train)) / np.std(y_train)
obsMatrix = np.column_stack((np.ones(len(x_train_normalized)), x_train_normalized))
obsMatrixT = np.transpose(obsMatrix)

# TODO: calculate theta using Batch Gradient Descent
gradient_theta = np.random.rand(2)

eta = 0.01
for i in range(10000):
    gradient_MSE = 2 / len(x_train_normalized) * np.dot(obsMatrixT, np.dot(obsMatrix, gradient_theta) - y_train_normalized)
    gradient_theta = gradient_theta - eta * gradient_MSE

print("training: "+MSE(x_train, y_train, gradient_theta))
print("test: "+MSE(x_test, y_test, gradient_theta))


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()