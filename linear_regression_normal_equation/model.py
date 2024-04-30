import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from global_name_space import ARGS


class LinearRegressionNormalEquation:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        self.feature_dim = self.x_train.shape[1]
        self.coeff = None

    @staticmethod
    def compute_design_matrix(x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    def compute_gradient(self, x, y, theta):
        A = self.compute_design_matrix(x)
        return 2 * A.T @ A @ theta - 2 * A.T @ y

    @staticmethod
    def compute_mean_squared_error(y_train, y_train_pred):
        upper = np.sum((y_train - y_train_pred)**2)
        lower = y_train.shape[0]
        return upper / lower

    @staticmethod
    def compute_r2_score(y_train, y_train_pred):
        upper = np.sum((y_train - y_train_pred)**2)
        lower = np.sum((y_train - y_train.mean())**2)
        return 1 - (upper / lower)

    @staticmethod
    def plot_y_and_y_pred(y, y_pred):
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(y.reshape((-1,)), y.reshape((-1,)), '--r', label="data")
        ax.plot(y.reshape((-1,)), y_pred.reshape((-1,)), 'o', label="pred")
        ax.legend()
        ax.set_xlabel("y")
        plt.show()

    def predict(self, x):
        A = self.compute_design_matrix(x)
        return A @ self.coeff

    def print_fit_measures(self, y, y_pred):
        print("mean squared error : ", self.compute_mean_squared_error(y, y_pred))
        print("R square           : ", self.compute_r2_score(y, y_pred))

    def train(self):
        A = self.compute_design_matrix(self.x_train)
        b = self.y_train
        self.coeff = la.inv(A.T @ A) @ (A.T @ b)