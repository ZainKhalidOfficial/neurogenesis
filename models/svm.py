import numpy as np

class SVM:
    """
    Only works for binary classification problems.
    Uses Gradient Descent to find the optimal hyperplane that separates two classes.
    """
    
    def __init__ (self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        _y = np.where(y <= 0, -1 , 1) # Convert labels to -1 and 1 because SVM predicts -1 or 1 for classes
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features) 
        self.b = 0
        
        # Gradient Descent
        
        # Hinge Loss: max(0, 1 - y_i * (w * x_i + b)) if the sample is correctly classified, the loss is 0 due to max(0 , ... ), otherwise it is positive
        # Also including Regularization term: lambda_param * ||w||^2, which penalizes large weights to avoid overfitting and encourage the hyperplane position at perfect center position between the two classes
        # So the loss function becomes:
        # Loss = lambda_param * ||w||^2 + 1/n_samples * sum(max(0, 1 - y_i * (w * x_i + b)))
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = _y[idx] * (np.dot(x_i, self.w) - self.b) >= 1 # true/false =? Check if the sample is correctly classified as equal signs (meaning same classes) result in positive (>= 1) value
                if condition:
                    # Update rule for correctly classified samples
                    self.w -= self.lr * (2 * self.lambda_param * self.w) # gradient descending for Regularization term only because the sample is already correctly classified
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, _y[idx]))
                    self.b -= self.lr * _y[idx]
                    
    def predict(self, X): # w * x - b = 0 where sign of the result determines the class 
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)    