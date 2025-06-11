import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load test dataset
# iris = datasets.load_iris()
# X, y= iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# # print(X_train.shape)
# # print(X_train[0])

# # Load the real dataset
# df = pd.read_csv('datasets/heart.csv')

# data = df.values

# X = data[:, :-1]  # Features
# y = data[:, -1]   # Target variable


# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# from models.knn import KNN
# clf = KNN(k=3)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# acc = np.sum(predictions == y_test) / len(y_test)
# print(f"Accuracy: {acc}")

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# #Plot
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color='blue', marker='o', s = 30)
# plt.show()

from models.linear_regression import LinearRegression
regressor = LinearRegression(lr=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, predictions)
print(f"Mean Squared Error: {mse_value}")

# Plotting the results
total_predictions = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color='blue', marker='o', s=30)
m2 = plt.scatter(X_test, y_test, color='red', marker='o', s=30)
plt.plot(X, total_predictions, color='green', linewidth=2)
plt.show()

