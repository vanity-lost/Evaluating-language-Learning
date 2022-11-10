import numpy as np
from sklearn.svm import SVR
from utils import *
from sklearn.metrics.pairwise import euclidean_distances

class S3VR():
    """
    The S3VR class used to do semi-surpervised regression tasks
    """
    def __init__(self, k_local, k_global, r, beta):
        """Initialization

        Args:
            k_local (int): k values for PLR_local
            k_global (int): k values for PLR_global
            r (float): probability threshold for choosing data pointsw
            beta (float): variance parameter
        """
        self.svr = None
        self.labeled_X = None
        self.unlabeled_X = None
        self.y = None
        self.dist = None

        self.k_local = k_local
        self.k_global = k_global
        self.r = r
        self.beta = beta

    def rbf(self, x1, x2, l=1):
        """Calulate the kernel mappings for x1 and x2

        Args:
            x1 (np.ndarray): the x1 with shape (n_1, d)
            x2 (np.ndarray): the x2 with shape (n_2, d)
            l (int, optional): the parameter of normal distribution. Defaults to 1.

        Returns:
            np.ndarray: the rbf kernel vectors
        """
        if ((x1-x2)**2).ndim > 1:
            return np.exp(-1 / (2 * (l**2)) * ((x1-x2)**2).sum(axis=1))
        else:
            return np.array([np.exp(-1 / (2 * (l**2)) * ((x1-x2)**2).sum())])

    def calc_distance(self):
        """Calculate the Euclidean distance and sort
        """
        self.dist = euclidean_distances(self.unlabeled_X, self.labeled_X)
        self.dist = np.argsort(self.dist, axis=1)

    def find_nn(self, index, k):
        """Find the k nearest neighbors of the point

        Args:
            point (np.ndarray): the data point with shape (d,)
            k (int): the number of nearest neighbors

        Raises:
            ValueError: the labeled X is not loaded

        Returns:
            np.ndarray: the matrix of the nearest neighbors with shape (k, d)
        """
        if self.labeled_X is None:
            raise ValueError("Not load datasets")
        if self.dist is None:
            self.calc_distance()
        index = self.dist[index, :k]
        return self.labeled_X[index], index

    def estimate_distribution(self, is_local):
        """Estimate the distribution

        Args:
            is_local (bool): whether use k_local or k_global

        Raises:
            ValueError: the datasets are not loaded

        Returns:
            np.ndarray, np.ndarray, int: the y_bar with shape (n_unlabeled,), the sigma_2_hat with shape (n_unlabeled,), the number of data points
        """
        if (self.labeled_X is None) or (self.unlabeled_X is None) or (self.y is None):
            raise ValueError("Not load datasets")

        k = self.k_local if is_local else self.k_global

        ones = np.ones((k, 1))
        num = self.unlabeled_X.shape[0]
        y_hat = np.zeros((num, 1))
        sigma_2_hat = np.zeros((num, 1))

        for i in range(num):
            nn, nn_index = self.find_nn(i, k)
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

    def data_generation(self):
        """Generate the combined training datasets

        Raises:
            ValueError: the datasets are not loaded

        Returns:
            np.ndarray, np.ndarray: the combined training X with shape (n_combined, d) and y with shape (n_combined,)
        """
        if (self.labeled_X is None) or (self.unlabeled_X is None) or (self.y is None):
            raise ValueError("Not load datasets")

        # estimate distribution
        y_local, sigma_2_local, n = self.estimate_distribution(True)
        y_global, sigma_2_global, _ = self.estimate_distribution(False)

        # avoid dividing by zero
        sigma_2_local[sigma_2_local == 0] = 1e-20
        sigma_2_global[sigma_2_global == 0] = 1e-20

        # conjugate
        # Equation (11), S3VR paper
        y_bar_conjugate = (y_global/sigma_2_global + n*y_local/sigma_2_local) / (1/sigma_2_global + n/sigma_2_local)
        sigma_2_conjugate = 1 / (1/sigma_2_global + n/sigma_2_local)

        # generate
        max_sigma_2, min_sigma_2 = sigma_2_conjugate.max(), sigma_2_conjugate.min()
        # Equation (12), S3VR paper
        pu = (sigma_2_conjugate - min_sigma_2) / (max_sigma_2 - min_sigma_2)
        X_hat = np.vstack((self.labeled_X.copy(), self.unlabeled_X[(pu >= self.r).reshape(-1,)]))
        y_hat = np.append(self.y.copy(), y_bar_conjugate[pu >= self.r])

        return X_hat, y_hat

    def fit(self, labeled_X, y, unlabeled_X):
        """Train the S3VR model on the datasets

        Args:
            labeled_X (np.ndarray): the labeled X with shape (n_labeled, d)
            y (np.ndarray): the labels with shape (n_labeled,) corresponding to the labeled X
            unlabeled_X (np.ndarray): the unlabeled X with shape (n_unlabeled, d)
        """
        self.labeled_X = labeled_X
        self.y = y
        self.unlabeled_X = unlabeled_X

        X_hat, y_hat = self.data_generation()
        self.svr = SVR(C=0.1)
        self.svr.fit(X_hat, y_hat)

    def predict(self, X):
        """Predict on X using the trained S3VR model

        Args:
            X (np.ndarray): the datasets to be predicted with shape (n_X, d)

        Raises:
            ValueError: the svr is not trained

        Returns:
            np.ndarray: the predictions of the dataset X with shape (n_X,)
        """
        if not self.svr:
            raise ValueError("No SVR is fitted.")

        return self.svr.predict(X)

