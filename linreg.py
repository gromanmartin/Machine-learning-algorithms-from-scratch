import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
a = np.random.rand(10, 1)
b = 1 + 2*a + np.random.rand(10, 1)


class LinearRegression:

    def __init__(self, X, y, sgd=False):
        self.X = X
        self.X0 = np.ones((len(X), 1))          # adding an intercept term
        self.X = np.hstack((self.X0, self.X))   # creating matrix which 1st column are the intercept terms
        self.y = y

    def fit(self):
        # (X^T X)^-1 X^T y
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), self.y)

    def predict(self):
        self.y_pred = np.dot(self.X, self.theta)
        return print(self.y_pred)


if __name__ == "__main__":
    linreg = LinearRegression(a, b)
    linreg.fit()
    linreg.predict()