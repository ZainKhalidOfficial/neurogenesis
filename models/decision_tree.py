import numpy as np
from collections import Counter

def entropy(y): 
    
    """ Calculate the entropy (uncertainity) of class labels.
        Entropy is a measure of the uncertainty in the data.
        
        Entropy = -Sum(p(x) * log2(p(x))) where p(x) is = y1 / len(y) for each class label y1
    """
    
    hist = np.bincount(y)
    ps = hist / len(y)
    
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature= feature
        self.threshold = threshold  
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None
    
    
class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_feats= None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
    def fit(self, X, y):
        self.n_feats= X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) #Take auto if not specified otherwise take the specified number of features but check if it is not more than the number of features in the data.
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0): # recursion
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        #stopping criteria
        if(n_samples < self.min_sample_split or depth >= self.max_depth or n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        #greedy search for the best split
        
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_indxs = self._split(X[:, best_feat], best_thresh)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_indxs, :], y[right_indxs], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)
        
        
        
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) 
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
            
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        """ Calculate the information gain of a split.
            IG = Entropy(parent) - (Weighted average of the entropy of the children)
            where parent is the current node and children are the nodes after the split.
        """
        
        # Calculate the parent entropy
        parent_entropy = entropy(y)
        
        # Generate Split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted average of the entropy of the children
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        
        # IG
        ig = parent_entropy - child_entropy
        return ig
        
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def predict(self, X):
        
        #traverse the tree and make predictions
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
    
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    def _most_common_label(self, y):
        counter = Counter(y)
        list_of_most_common = counter.most_common(1)
        return list_of_most_common[0][0] # 0 means first element, 0 again means the value from the tuple (value, count) 