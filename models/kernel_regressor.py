import numpy as np


class KernelRegressor:
    def __init__(self, kernel, lmd=1.0e-3):
        self.kernel = kernel
        self.lmd = lmd
        self.alpha = None

    def fit(self, X, y):
        self.X = X
        self.y = y

        K = self.kernel(X, X) + self.lmd * np.eye(len(X))

        # alpha = A^{-1} b : (N, 1)
        A = K  # (N, N)
        b = y  # (N,)
        self.alpha = np.linalg.solve(A, b)

        return self

    def predict(self, X):
        K = self.kernel(X, self.X)  # X: (M, D), self.X: (N, D), K: (M, D)
        y_hat = K @ self.alpha  # y_hat: (M, )

        return y_hat
