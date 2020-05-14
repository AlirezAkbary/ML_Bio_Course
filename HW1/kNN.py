import numpy as np


class kNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_tr, y_tr):
        self.X_tr = X_tr
        self.y_tr = y_tr

    def predict(self, X_te):
        num_test = X_te.shape[0]
        distance_matrix = np.sqrt(
            np.sum(X_te**2, axis=1).reshape((num_test,1)) + np.sum(self.X_tr**2, axis=1) - 2*X_te.dot(self.X_tr.T)
        )
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = self.y_tr[np.argsort(distance_matrix[i])[:self.k]].astype(np.int64)
            #print(closest_y)
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred

