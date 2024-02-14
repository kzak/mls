import numpy as np


class LinearRegressor:
    def __init__(self, alpha=1.0e-2, intercept=True):
        self.alpha = alpha
        self.intercept = intercept
        self.w = None

    def fit(self, X, y):
        # _X : # (N, D)
        _X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        _, D = _X.shape

        # _y : (N, 1)
        _y = y.reshape(-1, 1) if y.ndim == 1 else y

        # w = A^{-1} b : (D, 1)
        A = (_X.T @ _X) + self.alpha * np.eye(D)  # (D, D)
        b = _X.T @ _y  # (D, 1)
        self.w = np.linalg.solve(A, b)

        return self

    def predict(self, X):
        _X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        return (_X @ self.w).ravel()  # (N, )
