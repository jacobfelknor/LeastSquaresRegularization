"""
Custom implementation of LinearRegression
mimics the sklearn version
"""
import numpy as np

# Brute force find LT columns of a matrix
# def LI(M):
#     M = M.T
#     LI = [M[0]]
#     for i in range(len(M) - 1):
#         tmp = []
#         for r in LI:
#             tmp.append(r)
#         tmp.append(M[i])  # set tmp=LI+[M[i]]
#         if np.linalg.matrix_rank(tmp) > len(LI):  # test if M[i] is linearly independent from all (row) vectors in LI
#             LI.append(M[i])  # note that matrix_rank does not need to take in a square matrix
#     return np.array(LI).T


class LinearRegression:
    def __init__(self) -> None:
        self.weights = None
        self.RSS = None
        self.TSS = None
        self.R2 = None

    def fit(self, X: np.array, y: np.array):
        # calculate the weights for this model using normal equations
        self.weights = np.linalg.solve(X.T @ X, X.T @ y)
        # Using a built in...
        # self.weights = np.linalg.lstsq(X, y, rcond=None)[0]

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
