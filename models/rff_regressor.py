import numpy as np


class RffRegressor:
    def __init__(self, lmd=1.0e-3, n_rff=100, intercept=True):
        self.lmd = lmd
        self.n_rff = n_rff  # The number of random Fourier features
        self.intercept = intercept
        self.weights = None  # (n_rff, )

    def _rff(self, X):
        # (N, D) -> (N, n_rff) or (N, n_rff + 1)

        Omg_X = X @ self.Omg  # (N, D) @ (D, n_rff) -> (N, n_rff)
        Omg_X_b = Omg_X + self.b  # (N, n_rff)

        # Random Fourier features
        Phi_X = np.sqrt(2 / self.n_rff) * np.cos(Omg_X_b)
        if self.intercept:
            Phi_X = np.hstack([np.ones(len(X)).reshape(-1, 1), Phi_X])

        return Phi_X

    def fit(self, X, y):
        self.X = X
        self.y = y

        N, D = self.X.shape
        self.Omg = np.random.normal(loc=0, scale=1, size=(D, self.n_rff))  # (D, n_rff)
        self.b = np.random.uniform(0, 2 * np.pi, size=(1, self.n_rff))  # (1, n_rff)

        _y = y.reshape(-1, 1) if y.ndim == 1 else y  # (N, 1)
        Phi_X = self._rff(self.X)  # (N, n_rff)

        # Solve linear system: A x = b
        A = Phi_X.T @ Phi_X + self.lmd * np.eye(Phi_X.shape[1])  # (n_rff, n_rff)
        b = Phi_X.T @ _y  # (D, 1)
        self.weights = np.linalg.solve(A, b)  # (n_rff, )

        return self

    def predict(self, X):
        Phi_X = self._rff(X)  # (M, n_rff)
        y_hat = (Phi_X @ self.weights).ravel()  # (M, )
        return y_hat
