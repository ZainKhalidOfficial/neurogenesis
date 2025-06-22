import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))


# # Linear Regression Example

# # Load test dataset
#X, y = datasets.make_regression(n_samples=100, n_features=13, noise=10, random_state=42)

# # Load a real dataset
df = pd.read_csv('datasets/advertising_reg.csv')

data = df.values

X = data[:, :-1]  # Features
y = data[:, -1]   # Target variable

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from models.linear_regression import LinearRegression
regressor = LinearRegression(lr=0.00001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, predictions)
print(f"Mean Squared Error: {mse_value}")

#Individual prediction
individual_prediction = regressor.predict(np.array([[67.8,36.6,114]]))
print(f"Individual prediction: {individual_prediction}")

