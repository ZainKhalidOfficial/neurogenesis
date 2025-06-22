import numpy as np

class NaiveBayes:
    
    #No Innit because no parameters are needed for this model
    
    # Bayes Theorem:
    # P(y|X) = ( P(X|y) * P(y) ) / P(y)
    
    # X = (x1, x2, ..., xn) features
    # y = class label
    
    # Assuming features are independent given the class label, we can write it as:    
    # P(y|X) = ( P(x1,y) * P(x2|y) * ... * P(xn|y) * P(y) ) / P(X)
    
    # y = argmax_y P(x1,y) * P(x2|y) * ... * P(xn|y) * P(y) as each probability is b/w 0 and 1. Multiplying can result in very small values.
    # Taking log on both sides to avoid underflow:
    
    #Final form:
    # y = argmax_y log(P(x1,y)) + log(P(x2|y)) + ... + log(P(xn|y)) + log(P(y))
    
    # P(y) is the prior probability of class y
    # P(xi|y) is the likelihood of feature xi given class y called  the class conditional
    
    # Class Conditional Probability:
    # For continuous features, we can assume a Gaussian distribution:
    
    # Gaussian Distribution:
    # P(xi|y) = (1 / sqrt(2 * pi * var)) * exp(-(xi - mean)^2 / (2 * var))
    
    
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        self.n_classes = len(self._classes)
        
        # Initialize mean, variance, and prior for each class
        self._mean = np.zeros((self.n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((self.n_classes, n_features), dtype = np.float64)
        self._prior = np.zeros(self.n_classes, dtype = np.float64)
        
        # Calculate mean, variance, and prior for each class
        for c in self._classes:
            X_c = X[y == c]
            
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._prior[c] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            
            prior = np.log((self._prior[idx]))
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)]
            
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        return numerator / denominator
    
    