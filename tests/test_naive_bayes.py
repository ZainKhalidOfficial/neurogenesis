import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

### Naive Bayes Classifier Example

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

from models.naive_bayes import NaiveBayes

X, y =datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print(f"Accuracy: {accuracy(y_test, predictions)}")

