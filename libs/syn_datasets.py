import numpy as np


# ------------------------------
# 1D datasets
# ------------------------------
class SinDataset1D:
    def __init__(self, N=30, noise=0.2):
        self.X = np.random.uniform(low=-np.pi, high=np.pi, size=(N,))
        self.noise = noise
        self.y = self.true_fn(self.X) + noise * np.random.normal(scale=1.0, size=(N,))

    def true_fn(self, X):
        return np.sin(X)

    def load_data(self):
        return self.X, self.y


# ------------------------------
# 2D datasets
# ------------------------------
class SinExpDataset2D:
    def __init__(self, N=30, noise=0.2):
        self.X = np.random.uniform(low=-np.pi, high=np.pi, size=(N, 2))
        self.noise = noise
        self.y = self.true_fn(self.X) + noise * np.random.normal(scale=1.0, size=(N,))

    def true_fn(self, X):
        return np.sin(X[:, 0]) + (1 / (1 + np.exp(-X[:, 1])))

    def load_data(self):
        return self.X, self.y
