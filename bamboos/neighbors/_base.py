import numpy as np
from scipy.spatial import cKDTree


class CKDNearestNeibhors:
    def __init__(self, n_neighbors=1, radius=1, haversine=False):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.haversine = haversine

    def _to_cartesian(self, X):
        R = 3959  # radius of the Earth in miles

        lat = np.radians(X[:, 0])
        lng = np.radians(X[:, 1])

        x = np.cos(lat) * np.cos(lng)
        y = np.cos(lat) * np.sin(lng)
        z = np.sin(lat)

        return R * np.stack([x, y, z], axis=1)

    def _validate_X(self, X):
        if self.haversine:
            return self._to_cartesian(X)
        else:
            return X

    def fit(self, X):
        X = self._validate_X(X)
        self.ckdtree = cKDTree(X)

        return self

    def kneighbors(self, X, **kwargs):
        X = self._validate_X(X)
        return self.ckdtree.query(X, self.n_neighbors, **kwargs)

    def radius_neighbors(self, X, **kwargs):
        X = self._validate_X(X)
        return self.ckdtree.query_ball_point(X, self.radius, **kwargs)