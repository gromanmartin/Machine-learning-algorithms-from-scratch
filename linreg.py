import numpy as np
import matplotlib.pyplot as plt

# Random data as example
np.random.seed(0)
a = np.random.rand(10, 1)
b = 1 + 2*a + np.random.rand(10, 1)


class LinearRegression:

    def __init__(self, X, y, sgd=False, alpha=0.01, number_of_iter=10000):
        self.X = X
        self.X0 = np.ones((len(X), 1))          # adding an intercept term
        self.X = np.hstack((self.X0, self.X))   # creating matrix which 1st column are the intercept terms
        self.y = y
        self.theta = np.zeros((np.shape(self.X)[1], np.shape(self.y)[1]))
        self.sgd = sgd
        self.alpha = alpha
        self.number_of_iter = number_of_iter
        self.cost = np.zeros((self.number_of_iter, 1))

    def fit(self):
        if self.sgd:
            # Batch gradient descent training
            i = 0
            while i < self.number_of_iter:                              # until convergence is reached
                h_x = np.dot(self.X, self.theta)                        # result is matrix (10,1)
                error = np.dot(np.transpose(self.X), self.y - h_x)      # result is matrix (2,1)
                self.theta = self.theta + self.alpha * error            # result is matrix (2,1)
                self.cost[i] = 0.5*np.sum(np.square(self.y - h_x))
                i = i+1
        else:
            # Normal equations training
            # (X^T X)^-1 X^T y
            self.theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), self.y)

    def predict(self):
        self.y_pred = np.dot(self.X, self.theta)
        return print(self.y_pred)

    def cost_plot(self):
        # function plotting the cost function
        x_labels = np.arange(0, self.number_of_iter, 1)
        plt.plot(x_labels, self.cost)
        plt.axis([0, self.number_of_iter, 0, 1])
        plt.show()


if __name__ == "__main__":
    linreg = LinearRegression(a, b, sgd=True)
    linreg.fit()
    linreg.predict()
    linreg.cost_plot()
