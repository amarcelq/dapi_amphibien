from abc import ABC, abstractmethod
from typing import Optional
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
    def __init__(self, n_dims: Optional[int] = None, variance_cutoff: Optional[float] = 0.98):
        self.scaler = preprocessing.StandardScaler()
        self.variance_cutoff = variance_cutoff

        if n_dims is None and variance_cutoff is None:
            raise ValueError("You have to provide n_dims or variance.")

        if n_dims is not None:
            self.pca = decomposition.PCA(n_components=n_dims)
        else:
            self.pca = decomposition.PCA(n_components=variance_cutoff)

    def __call__(self, x):
        x = self.scaler.fit_transform(x)
        return self.pca.fit_transform(x)

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