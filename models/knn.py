import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__ (self, k=3):
        
        self.k = k
        
    def fit(self, X, y): # No training needed in KNN Algorithm
        self.X_train = X
        self.y_train = y
    
    def predict(self, X): # Recieve multiple samples in capitalized "X"
        predicted_labels = [self._predict(x) for x in X] #List of all predictions for each sample in "X"
        return np.array(predicted_labels)
        
    def _predict(self, x): # Recieve single sample in lowercase "x"
        
        # Compute distances between x input sample and all X_train samples 
        distances = [ euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k] #Sort distances and get indices of the k number of smallest distances
        k_nearest_labels = [self.y_train[i] for i in k_indices] #Convert the k_nearest indices to labels
        
        # Return the most common class label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1) # 1 means we want the most common label, 2 means the 1st + 2nd most common label, etc.
        
        return most_common[0][0] # Return the most common label, which is the first element of the tuple in most_common list
    
    
        
        