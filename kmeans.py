import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.7)


class KMeans:

    def __init__(self, x, k, steps=20, interactive_plotting=True):
        self.x = x
        self.k = k
        self.number_of_examples = np.shape(self.x)[0]
        self.centers = self.init_centers()
        self.steps = steps
        self.interactive_plotting = interactive_plotting

    def fit(self):
        for step in range(0, self.steps):
            """
            1. part  - find the center closest to each example
            """
            c = []
            self.c_indexes = []
            for i in range(0, self.number_of_examples):
                help_vector = np.zeros((self.k, ))
                for j in range(0, self.k):
                    help_vector[j] = np.square(np.linalg.norm(self.x[i] - self.centers[j]))
                ind = np.argmin(help_vector)
                c.append(self.centers[ind])
                self.c_indexes.append(ind)

            """
            2. part - update the cluster centers
            """
            c = np.asarray(c)
            self.c_indexes = np.asarray(self.c_indexes)
            for j in range(0, self.k):
                indexes = np.where(self.c_indexes == j)
                new_center = np.sum(c[indexes] * self.x[indexes], axis=0) / np.sum(c[indexes], axis=0)
                self.centers[j] = new_center

            if self.interactive_plotting:
                plt.ion()
                clrs = get_cmap('tab10').colors
                for i in range(0, self.k):
                    idx = np.where(self.c_indexes == i)
                    plt.scatter(self.x[idx, 0], self.x[idx, 1], c=clrs[i], alpha=0.3)
                    plt.scatter(self.centers[i, 0], self.centers[i, 1], c=clrs[i], marker='x', s=60)
                plt.draw()
                plt.pause(0.2)
                plt.clf()

    def init_centers(self):
        # pick k random examples and set them as cluster centers
        random_indexes = np.random.randint(0, self.number_of_examples, size=self.k) # randomly select k indexes
        random_centers = self.x[random_indexes]

        # or just initiate randomly
        random_centers2 = np.random.rand(self.k, np.shape(self.x)[1])

        return random_centers

    def plot_results(self):
        plt.ioff()
        clrs = get_cmap('tab10').colors
        for i in range(0, self.k):
            idx = np.where(self.c_indexes == i)
            plt.scatter(self.x[idx, 0], self.x[idx, 1], c=clrs[i], alpha=0.3)
            plt.scatter(self.centers[i, 0], self.centers[i, 1], c=clrs[i], marker='x', s=60)
            plt.text(self.centers[i, 0], self.centers[i, 1], str(np.around(self.centers[i, :], 2)))
        plt.show()


if __name__ == "__main__":
    kmeans = KMeans(X, 4)
    kmeans.fit()
    kmeans.plot_results()