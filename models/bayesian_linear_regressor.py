from pdb import set_trace

import numpy as np

# see: https://zenn.dev/zak2718/articles/50c8231d7891ce


class BayesianLinearRegressor:
    def __init__(
        self,
        var_y: float = 1.0e-3,
        var_w: float = 1.0e-3,
        intercept=True,
    ):
        self.var_y = var_y  # \sigma_y^2
        self.var_w = var_w  # \sigma_w^2
        self.intercept = intercept

        self.mu_w = None
        self.Sgm_w = None

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y

        _X = self._add_intercept(X)
        _, D = _X.shape

        self.Sgm_w_inv = (1 / self.var_y) * _X.T @ _X + 1 / self.var_w * np.eye(D)
        self.Sgm_w = np.linalg.inv(self.Sgm_w_inv)
        self.mu_w = (1 / self.var_y) * self.Sgm_w @ _X.T @ y

        return self

    def predict(
        self,
        X_new,
        return_var=False,
    ):
        _X_new = self._add_intercept(X_new)
        M, D = _X_new.shape

        mu_y = _X_new @ self.mu_w

        if not return_var:
            return mu_y

        Sgm_y = _X_new @ self.Sgm_w @ _X_new.T + self.var_y * np.eye(M)

        return mu_y, Sgm_y

    def _add_intercept(self, X):
        X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        return X
