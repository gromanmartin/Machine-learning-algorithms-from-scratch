import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
A = np.array([[1, 2], [3, 4], [5, 6]])
A = [[7,4,3],[4,1,8],[6,3,5],[8,6,1],[8,5,7],[7,2,9],[5,3,3],[9,5,8],[7,4,5],[8,2,2]]


class PCA:
    """
    Principal component analysis using standardized data
    """
    def __init__(self, x, n_components):
        self.scaler = StandardScaler(with_std=False)
        self.x = self.standardize(x)
        self.n_components = n_components

    def fit(self):
        sigma = np.cov(self.x.T)
        eigvalues, eigvectors = np.linalg.eig(sigma)
        zipped = list(zip(eigvalues, eigvectors))
        eigvalues_sorted = [x[0] for x in sorted(zipped, key=lambda x: x[0], reverse=True)] # Sort eigvalues from the largest to the smallest
        eigvectors_sorted = [x[1] for x in sorted(zipped, key=lambda x: x[0], reverse=True)]
        var_explained = [eigvalues_sorted[i]/np.sum(eigvalues_sorted) for i in range(len(eigvalues_sorted))][:self.n_components]
        for i, var_of_pcomp in enumerate(var_explained):
            print('PC{} carries {}% of variance in data.'.format(i, var_of_pcomp*100))
        return np.asarray(eigvectors_sorted[:self.n_components])

    def transform(self, eigvecs):
        transformed_data = eigvecs.dot(self.x.T)
        print(transformed_data.T)

    def standardize(self, data):
        return self.scaler.fit_transform(data)


if __name__ == '__main__':
    pca = PCA(A, 2)
    u = pca.fit()
    pca.transform(u)
