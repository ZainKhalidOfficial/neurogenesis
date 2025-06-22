import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))



# Logistic Regression Example

# # Load the dataset
# dataset = datasets.load_breast_cancer()
# X, y = dataset.data, dataset.target

# Load a real dataset
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
