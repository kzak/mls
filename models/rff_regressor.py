import numpy as np


class RffRegressor:
    def __init__(self, lmd=1.0e-3, n_rff=30):
        self.lmd = lmd
        self.n_rff = n_rff  # The number of random Fourier features
        self.alpha = None  # (n_rff, )

    def _rff(self, X):
        # (N, D) -> (N, n_rff)
        Wx = X @ self.Omg  # (N, D) @ (D, n_rff) -> (N, n_rff)
        b = self.b  # (1, n_rff) to be broadcasted to (N, n_rff)
        Wx_b = Wx + b  # (N, n_rff)
        X_rff = np.sqrt(2 / self.n_rff) * np.cos(Wx_b)
        return X_rff

    # Optimize in weight space view
    def fit(self, X, y):
        self.X = X
        self.y = y

        N, D = self.X.shape
        self.Omg = np.random.normal(loc=0, scale=1, size=(D, self.n_rff))  # (D, n_rff)
        self.b = np.random.uniform(0, 2 * np.pi, size=(1, self.n_rff))  # (1, n_rff)

        _y = y.reshape(-1, 1) if y.ndim == 1 else y  # (N, 1)
        _X = self._rff(self.X)  # (N, n_rff)

        A = (_X.T @ _X) + self.lmd * np.eye(self.n_rff)  # (n_rff, n_rff)
        b = _X.T @ _y  # (D, 1)
        self.alpha = np.linalg.solve(A, b)  # (n_rff, )

        return self

    def predict(self, X):
        _X = self._rff(X)  # (M, n_rff)
        y_hat = (_X @ self.alpha).ravel()  # (M, )
        return y_hat
