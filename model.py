import numpy as np
from sklearn.svm import SVR
from utils import *

class S3VR():
    def __init__(self, k_local, k_global, r, beta):
        self.svr = None
        self.k_local = k_local
        self.k_global = k_global
        self.r = r
        self.beta = beta
        return

    def rbf(self, x1, x2, l=1):
        if ((x1-x2)**2).ndim > 1:
            return np.exp(-1 / (2 * (l**2)) * ((x1-x2)**2).sum(axis=1))
        else:
            return np.array([np.exp(-1 / (2 * (l**2)) * ((x1-x2)**2).sum())])

    def find_nn(self, point, k):
        dist = np.linalg.norm(self.labeled_X - point, axis=1)
        index = np.argsort(dist)[:k]
        return self.labeled_X[index], index

    def estimate_distribution(self, is_local):
        k = self.k_local if is_local else self.k_global

        ones = np.ones((k, 1))
        num = self.unlabeled_X.shape[0]
        y_hat = np.zeros((num, 1))
        sigma_2_hat = np.zeros((num, 1))

        for i in range(num):
            nn, nn_index = self.find_nn(self.unlabeled_X[i], k)
            k_star = self.rbf(self.unlabeled_X[i], nn).reshape(-1, 1)
            K_hat = self.rbf(nn, nn) - ones @ k_star.T - k_star @ ones.T + self.rbf(self.unlabeled_X[i], self.unlabeled_X[i]) * ones @ ones.T
            # Equation (15), PLR paper
            cov = (self.beta * K_hat + np.identity(k))
            # Equation (14), PLR paper
            mu = cov @ (ones/k)
            y_bar_nn = self.y[nn_index].sum(axis=0) / k
            diff = self.y[nn_index].reshape(-1, 1) - y_bar_nn * ones
            y_hat[i] = y_bar_nn + mu.T @ diff
            sigma_2_hat[i] = diff.T @ cov @ diff / k

        return y_hat, sigma_2_hat, num

    def data_generation(self, labeled_X, y, unlabeled_X):
        self.labeled_X = labeled_X
        self.y = y
        self.unlabeled_X = unlabeled_X
        # estimate distribution
        y_local, sigma_2_local, n = self.estimate_distribution(True)
        y_global, sigma_2_global, _ = self.estimate_distribution(False)

        # avoid dividing by zero
        sigma_2_local[sigma_2_local == 0] = 1e-20

        # conjugate
        # Equation (11), S3VR paper
        y_bar_conjugate = (y_global/sigma_2_global + n*y_local/sigma_2_local) / (1/sigma_2_global + n/sigma_2_local)
        sigma_2_conjugate = 1 / (1/sigma_2_global + n/sigma_2_local)

        # generate
        max_sigma_2, min_sigma_2 = sigma_2_conjugate.max(), sigma_2_conjugate.min()
        # Equation (12), S3VR paper
        pu = (sigma_2_conjugate - min_sigma_2) / (max_sigma_2 - min_sigma_2)
        X_hat = np.vstack((self.labeled_X.copy(), self.unlabeled_X[(pu >= self.r).reshape(-1,)]))
        y_hat = np.append(y.copy(), y_bar_conjugate[pu >= self.r])

        return X_hat, y_hat

    def fit(self, labeled_X, y, unlabeled_X):
        X_hat, y_hat = self.data_generation(labeled_X, y, unlabeled_X)
        self.svr = SVR()
        self.svr.fit(X_hat, y_hat)
        print(f'The training rmse is {RMSE(y_hat, self.svr.predict(X_hat))}')

    def predict(self, X):
        if not self.svr:
            raise ValueError("No SVR is fitted.")

        return self.svr.predict(X)

