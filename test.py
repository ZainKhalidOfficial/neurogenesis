import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load test dataset
# iris = datasets.load_iris()
# X, y= iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# # print(X_train.shape)
# # print(X_train[0])

# Load the real dataset
df = pd.read_csv('datasets/heart.csv')

data = df.values

X = data[:, :-1]  # Features
y = data[:, -1]   # Target variable


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


from knn import KNN
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {acc}")