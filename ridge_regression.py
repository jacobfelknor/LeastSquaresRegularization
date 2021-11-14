"""
Custom implementation of RidgeRegression
mimics the sklearn version
"""

import numpy as np


class Ridge:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.weights = None
        self.RSS = None
        self.TSS = None
        self.R2 = None

    def fit(self, X: np.array, y: np.array):
        # calculate the weights for this model using normal equations
        self.weights = np.linalg.solve((X.T @ X) + self.alpha * np.identity(X.shape[1]), X.T @ y)

        # for compatibility with sklearn's Ridge
        self.coef_ = self.weights

    def predict(self, X: np.array) -> np.array:
        # evaluate fitted model at these X
        return X.dot(self.weights)

    def score(self, X: np.array, y_true: np.array) -> float:
        # get the prediction for these X
        pred = self.predict(X)

        # Sum of squares residuals (residual sum of squares)
        self.RSS = np.sum((y_true - pred) ** 2)
        # Total sum of squares
        self.TSS = np.sum((y_true - np.mean(y_true)) ** 2)
        # R^2 = 1 - RSS/TSS
        self.R2 = 1 - self.RSS / self.TSS

        return self.R2
