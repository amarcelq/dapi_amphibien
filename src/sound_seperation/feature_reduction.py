from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from sklearn import preprocessing, decomposition, manifold
import umap

class FeatureReductionMethod(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

# good for linear
class PCA(FeatureReductionMethod):
    def __init__(self, n_dims: Optional[int] = None, variance_cutoff: float = 0.98, x=None):
        self.scaler = preprocessing.StandardScaler()
        self.variance_cutoff = variance_cutoff
        self.x = x

        if n_dims is None:
            if x is None:
                raise ValueError("Data 'x' must be provided to compute number of dimensions.")
            self.x_scaled = self.scaler.fit_transform(x)
            n_dims = self._calc_n_dims(self.x_scaled)
            print(f"Calculated n_dims: {n_dims}")
        else:
            self.x_scaled = self.scaler.fit_transform(x) if x is not None else None

        self.pca = decomposition.PCA(n_components=n_dims)

    def __call__(self, x=None):
        if x is None:
            if self.x_scaled is None:
                raise ValueError("No data provided and no internal data available.")
            return self.pca.fit_transform(self.x_scaled)
        x = self.scaler.fit_transform(x)
        return self.pca.fit_transform(x)

    def _calc_n_dims(self, x_scaled):
        pca = decomposition.PCA()
        pca.fit(x_scaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        return np.argmax(cumulative_variance >= self.variance_cutoff) + 1

# good for non linear
class UMAP(FeatureReductionMethod):
    def __init__(self, n_dims: int, n_neighbors=15, min_dist=0.1):
        self.scaler = preprocessing.StandardScaler()
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap = umap.UMAP(n_components=n_dims, n_neighbors=self.n_neighbors, min_dist=self.min_dist)

    def __call__(self, x):
        x = self.scaler.fit_transform(x)
        return self.umap.fit_transform(x)

class TSNE(FeatureReductionMethod):
    def __init__(self, n_dims: int, perplexity=30, learning_rate=200, n_iter=1000):
        self.scaler = preprocessing.StandardScaler()
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tsne = manifold.TSNE(n_components=n_dims, perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)

    def __call__(self, x):
        x = self.scaler.fit_transform(x)
        return self.tsne.fit_transform(x)
# TODO: Maybe use a Autoencoder and train it with the Sound Speration Model to do both in once
# TODO: Also here really extensive and therefore first give the classical method a shoot and if
# TODO: its really bad try the autoencoder apporach