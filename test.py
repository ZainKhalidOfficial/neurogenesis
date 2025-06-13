import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# # KNN Classifier Example

# # Load test dataset
# iris = datasets.load_iris()
# X, y= iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# # Load a real dataset
# df = pd.read_csv('datasets/heart_binclf.csv')# advertising_reg

# data = df.values

# X = data[:, :-1]  # Features
# y = data[:, -1]   # Target variable

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# # KNN Classifier Example

# from models.knn import KNN
# clf = KNN(k=3)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# acc = np.sum(predictions == y_test) / len(y_test)
# print(f"Accuracy: {acc}")

# # Linear Regression Example

# # Load test dataset
# X, y = datasets.make_regression(n_samples=100, n_features=13, noise=10, random_state=42)

# # Load a real dataset
# df = pd.read_csv('datasets/advertising_reg.csv')

# data = df.values

# X = data[:, :-1]  # Features
# y = data[:, -1]   # Target variable

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# from models.linear_regression import LinearRegression
# regressor = LinearRegression(lr=0.00001, n_iters=1000)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)

# def mse(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# mse_value = mse(y_test, predictions)
# print(f"Mean Squared Error: {mse_value}")

# #Individual prediction
# individual_prediction = regressor.predict(np.array([[67.8,36.6,114]]))
# print(f"Individual prediction: {individual_prediction}")


# Logistic Regression Example

# # Load the dataset
# dataset = datasets.load_breast_cancer()
# X, y = dataset.data, dataset.target

# # Load a real dataset
df = pd.read_csv('datasets/heart_binclf.csv') #advertising_reg
data = df.values
X = data[:, :-1]  # Features
y = data[:, -1]   # Target variable
X = StandardScaler().fit_transform(X)  # Standardize features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from models.logistic_regression import LogisticRegression

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = LogisticRegression(lr=0.001, n_iters=1000)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)   

print(f"Accuracy: {accuracy(y_test, predictions)}")


