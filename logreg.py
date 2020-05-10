import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from celluloid import Camera

df = pd.read_csv("https://raw.githubusercontent.com/nikhilkumarsingh/Machine-Learning-Samples/master/Logistic_Regression/dataset1.csv")
x = df.iloc[:,:2]
y = df.iloc[:,2:]
x = x.to_numpy()
y = np.ravel(y.to_numpy())


class LogisticRegression:

    def __init__(self, X, y, alpha=0.01, number_of_iter=100):
        self.X = X
        self.X0 = np.ones((len(X), 1))                              # adding an intercept term
        self.X = np.hstack((self.X0, self.X))
        self.y = y
        self.theta = np.zeros((np.shape(self.X)[1], ))
        self.alpha = alpha
        self.number_of_iter = number_of_iter
        self.thetas = []

    def fit(self):
        # Batch gradient descent training
        i = 0
        while i < self.number_of_iter:                              # until convergence is reached
            h_x = 1/(1 + np.exp(-np.dot(self.X, self.theta)))       # result is matrix (99,)
            error = np.dot(np.transpose(self.X), self.y - h_x)      # result is matrix (3,)
            self.theta = self.theta + self.alpha * error            # result is matrix (3,)
            i = i+1
            self.thetas.append(self.theta)

    def predict(self):
        self.y_pred = 1/(1 + np.exp(-np.dot(self.X, self.theta)))
        for i in range(0, len(self.y_pred)):
            if self.y_pred[i] >= 0.5:
                self.y_pred[i] = 1
            else:
                self.y_pred[i] = 0
        return self.y_pred

    def interactive_plotting(self):
        # Function visualizing the process of training
        plt.ion()
        for i in range(0, len(self.thetas)):
            self.y_pred = 1 / (1 + np.exp(-np.dot(self.X, self.thetas[i])))
            for j in range(0, len(self.y_pred)):
                if self.y_pred[j] >= 0.5:
                    self.y_pred[j] = 1
                else:
                    self.y_pred[j] = 0
            plt.scatter(x[:, 0], x[:, 1], c=self.y_pred)
            plt.axis([0, 10, 0, 10])
            x_db = np.arange(0, 10, 1)
            y_db = - (self.thetas[i][0] + self.thetas[i][1] * x_db) / self.thetas[i][2]
            plt.plot(x_db, y_db, c='red', label='Decision boundary')
            plt.draw()
            plt.pause(0.5)
            plt.clf()

    def plot_results(self):
        # The result after training is complete
        plt.scatter(x[:, 0], x[:, 1], c=self.y_pred)
        plt.axis([0, 10, 0, 10])
        x_db = np.arange(0, 10, 1)
        y_db = - (self.theta[0] + self.theta[1] * x_db) / self.theta[2]
        plt.plot(x_db, y_db, c='red', label='Decision boundary')
        plt.show()


if __name__ == "__main__":
    model = LogisticRegression(x, y)
    model.fit()
    # model.interactive_plotting()
    model.predict()
    model.plot_results()
