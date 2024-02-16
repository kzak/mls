import numpy as np


class LogisticRegressor:
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.X = None  # (N, D)
        self.y = None  # (N, 1)
        self.w = None  # (D, 1)
        self.loglik_list = []

    def fit(self, X, y, n_iter=100, eta=1.0e-3, lmd=1.0e-2):
        self.X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        _, D = self.X.shape

        self.y = y.reshape(-1, 1) if y.ndim == 1 else y

        w = np.random.normal(scale=0.01, size=(D, 1))
        self.loglik_list.append(loglik(w, self.X, self.y))
        for i in range(n_iter):
            grad_w = grad(w, self.X, self.y, lmd)
            # Gradient ascending method because we're maximizing a log likelihood.
            w = w + eta * grad_w
            self.loglik_list.append(loglik(w, self.X, self.y))

        self.w = w

        return self

    def predict_proba(self, X):
        _X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        return (_X @ self.w).ravel()  # (N, )

    def predict_label(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def sigmoid(x, clipping=True):
    if clipping:
        x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))


def loglik(w, X, y):
    f = sigmoid(X @ w)
    return (y * np.log(f) + (1 - y) * np.log(1 - f)).sum()


def grad(w, X, y, lmd):
    f = sigmoid(X @ w)
    grad_w = ((y - f) * X).sum(axis=0).reshape(-1, 1)
    return grad_w
