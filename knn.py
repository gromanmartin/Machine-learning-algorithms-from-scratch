import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import make_blobs

# iris = datasets.load_iris()
# x = iris.data[:, :2]    # we only take the first two features.
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=250)
centers = [[1, 1], [-1, -1], [1, -1]]
xx, yy = make_blobs(n_samples=100, centers=centers, cluster_std=0.7)
X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.1, shuffle=True, random_state=250)


class KNN:

    def __init__(self, train_x, test_x, train_y, test_y,  k, interactive_plotting=True):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.k = k
        self.interactive_plotting = interactive_plotting

    def fit(self):
        pass

    def predict(self):
        for data_point in self.test_x:
            distances = self.eucl_dist(data_point, self.train_x)
            pred_y, min_idxs = self.classifier(distances, self.k)
            neighbours = self.train_x[min_idxs]
            if self.interactive_plotting:
                plt.ion()
                plt.set_cmap('jet')
                plt.scatter(self.train_x[:, 0], self.train_x[:, 1], alpha=0.8, c=self.train_y, marker='o', s=15)
                plt.scatter(data_point[0], data_point[1], alpha=1, marker='o', s=30, c=pred_y)
                for neighbour in neighbours:
                    x_data = np.array([data_point[0], neighbour[0]])
                    y_data = np.array([data_point[1], neighbour[1]])
                    plt.plot(x_data, y_data, c='r', linestyle='--', linewidth=1)
                plt.draw()
                plt.pause(1)
                plt.clf()

    def classifier(self, data, number_of_neighbours):
        # Function deciding the class of provided example, considering ties
        # data = euclidean distances from selected point to other ones
        min_indexes = data.argsort()[:number_of_neighbours]
        predicted_y = self.train_y[min_indexes]
        vals = list(Counter(predicted_y).keys())
        counts = list(Counter(predicted_y).values())  # counts the elements' frequency
        zipped = zip(vals, counts)
        zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
        if len(zipped) > 1 and zipped[1][0] == zipped[1][1]:
            # if there is equal amount of different classes in the neighbourhood, decrease the number of neighbours
            self.classifier(data, number_of_neighbours - 1)
            print('Decreasing number of neighbours to {}'.format(number_of_neighbours-1))
        return zipped[0][0], min_indexes


    def eucl_dist(self, point, data):
        # Calculates Euclidean distances from selected point to every other point in data
        # data - training dataset
        eucl_distances = []
        for other_point in data:
            dist = np.linalg.norm(other_point - point)
            eucl_distances.append(dist)
        return np.asarray(eucl_distances)


if __name__ == "__main__":
    knn = KNN(X_train, X_test, y_train, y_test, 5)
    knn.predict()
