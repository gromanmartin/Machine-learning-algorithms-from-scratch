import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
A = np.array([[1, 2], [3, 4], [5, 6]])


class PCA:

    def __init__(self, x, n_components):
        self.scaler = StandardScaler()
        self.x = self.standardize(x)
        self.n_components = n_components

    def fit(self):
        sigma = 0
        for i, example in enumerate(self.x):
            example = example.reshape(-1, 1)
            mult = example * np.transpose(example)
            sigma = np.add(sigma, mult)
        _, u = np.linalg.eig(sigma)
        return u

    def transform(self):
        pass

    def standardize(self, data):
        return self.scaler.fit_transform(data)


if __name__ == '__main__':
    pca = PCA(A, 2)
    print(pca.fit())