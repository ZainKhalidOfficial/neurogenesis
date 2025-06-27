import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

from models.svm import SVM

X, y = datasets.make_blobs(n_samples=1000, n_features=10, centers=2, cluster_std=1.05, random_state=42)
y = np.where(y == 0, -1, 1) # Convert labels to -1 and 1 for SVM


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVM()
clf.fit(X, y)

accuracy = np.sum(clf.predict(X_test) == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")

