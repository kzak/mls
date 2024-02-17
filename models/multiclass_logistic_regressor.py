from pdb import set_trace

import numpy as np


class MulticlassLogisticRegressor:
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.X = None  # (N, D)
        self.y = None  # (N, 1)
        self.W = None  # (D, C)
        self.loglik_list = []

    def fit(self, X, y, n_iter=100, eta=1.0e-3, lmd=1.0e-2):
        self.X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        N, D = self.X.shape

        self.y = y  # (N, )
        self.C = len(set(y))

        self.T = np.zeros(shape=(N, self.C))  # (N, C)
        for n, y_n in enumerate(y):
            self.T[n, y_n] = 1

        W = np.random.normal(scale=0.01, size=(D, self.C))
        self.loglik_list.append(loglik(W, self.X, self.T))
        for i in range(n_iter):
            grad_W = grad(W, self.X, self.T, lmd)
            # Gradient ascending method because we're maximizing a log likelihood.
            W = W + eta * grad_W
            self.loglik_list.append(loglik(W, self.X, self.T))

        self.W = W

        return self

    def predict_proba(self, X):
        _X = np.hstack([np.ones(len(X)).reshape(-1, 1), X]) if self.intercept else X
        return _X @ self.W  # (N, C)

    def predict(self, X):
        proba = self.predict_proba(X)  # (N, C)
        classes = proba.argmax(axis=1)  # (N, )
        return classes


def softmax(X, clipping=True):
    if clipping:
        X = np.clip(X, -10, 10)  # (N, C)

    e_X = np.exp(X)  # (N, C)
    e_X_sum = e_X.sum(axis=1).reshape(-1, 1)  # (N, 1)

    return e_X / e_X_sum  # (N,C)


def loglik(W, X, T):
    P = softmax(X @ W)
    return (T * P).sum()


def grad(W, X, T, lmd):
    P = softmax(X @ W)  # (N, C)
    grad_W = X.T @ (T - P)  # (D, C)
    return grad_W + lmd * W
