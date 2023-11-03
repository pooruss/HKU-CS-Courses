import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
np.random.seed(1)

class BaseAlgorithm(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        return super().__str__()
    
    def save_params(self, file_path):
        params = {attr: getattr(self, attr) for attr in self.__dict__ if not attr.startswith('_')}
        for key in params:
            if isinstance(params[key], np.ndarray):
                params[key] = params[key].tolist()
        json.dump(params, open(file_path, "w"), indent=2)

# SimpleSVM class
class SimpleSVM(BaseAlgorithm):
    """
    A simple implementation of a Support Vector Machine using gradient descent.
    """
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        super().__init__()
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize parameters to zero
        self.w = np.zeros(n_features)
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w)) >= 1
                if condition:
                    # If correctly classified, only regularize w
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If incorrectly classified, update w
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
    
    def predict(self, X):
        linear_output = np.dot(X, self.w)
        return np.sign(linear_output)
    
    def __str__(self) -> str:
        return "SimpleSVM"
    

class AdaBoost(BaseAlgorithm):
    def __init__(self, n_clf=5):
        super().__init__()
        self.n_clf = n_clf
        self.alphas = []
        self.stumps = {}

    def fit(self, X, y):
        """
        Fit the model using the training data.
        """
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        # Initialize variables to store the stumps and alphas
        self.stumps = np.zeros((self.n_clf, 3))
        self.alphas = np.zeros(self.n_clf)

        # Convert labels to -1 and 1
        y = np.where(y == 0, -1, 1)

        # Train the stumps
        for clf_idx in range(self.n_clf):
            stump = np.zeros(3)
            min_error = float('inf')

            # Find the best threshold and feature for a stump
            for feature_idx in range(n_features):
                feature_values = np.sort(np.unique(X[:, feature_idx]))
                thresholds = (feature_values[:-1] + feature_values[1:]) / 2.0
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        pred = np.ones(n_samples)
                        pred[polarity * X[:, feature_idx] < polarity * threshold] = -1
                        error = sum(w[y != pred])

                        # Store the stump if the error is less than the previous minimum
                        if error < min_error:
                            min_error = error
                            stump = np.array([feature_idx, threshold, polarity])

            # Calculate the alpha value for the stump
            eps = 1e-10  # to prevent division by zero
            stump_error = min_error
            alpha = 0.5 * np.log((1 - stump_error + eps) / (stump_error + eps))
            self.alphas[clf_idx] = alpha
            self.stumps[clf_idx] = stump

            # Update weights
            stump_pred = np.ones(n_samples)
            stump_pred[stump[2] * X[:, int(stump[0])] < stump[2] * stump[1]] = -1
            w *= np.exp(-alpha * y * stump_pred)
            w /= np.sum(w)  # Normalize to 1

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        """
        n_samples = X.shape[0]
        clf_preds = np.zeros((self.n_clf, n_samples))
        for t in range(self.n_clf):
            stump = self.stumps[t]
            predictions = np.ones(n_samples)
            predictions[stump[2] * X[:, int(stump[0])] < stump[2] * stump[1]] = -1
            clf_preds[t] = predictions
        y_pred = np.dot(self.alphas, clf_preds)
        y_pred = np.sign(y_pred)
        return y_pred
    
    def __str__(self) -> str:
        return "AdaBoost"