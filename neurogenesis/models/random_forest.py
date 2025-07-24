import numpy as np
from .decision_tree import DecisionTree
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)    
    
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    list_of_most_common = counter.most_common(1)
    return list_of_most_common[0][0] # 0 means first element, 0 again means the value from the tuple (value, count) 

class RandomForest:
    def __init__(self, n_trees=100, min_sample_split=2, max_depth=100, n_feats= None):
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTree(min_sample_split=self.min_sample_split, max_depth=self.max_depth, n_feats=self.n_feats)
            
            X_samples, y_samples = bootstrap_sample(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([ tree.predict(X) for tree in self.trees])
        
        # Given 3 trees and 4 samples for each tree the predictions can look like this:
        # [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]] where each row corresponds to a tree's predictions.
        # But for majority voting we need same sample from each tree so we transpose it:
        
        tree_preds = np.swapaxes(tree_preds, 0, 1) # swaping axes 0 & 1  
        # Now it's like: [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]]
        
        y_pred = [ most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)     