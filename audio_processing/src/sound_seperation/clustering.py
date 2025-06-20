from abc import ABC, abstractmethod
from sklearn import cluster, mixture

class ClusteringMethod(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

class AHC(ClusteringMethod):
    def __init__(self, n_clusters: int):
        self.ahc = cluster.AgglomerativeClustering(n_clusters=n_clusters)

    def __call__(self, x):
        return self.ahc.fit_predict(x)

class DBSCAN(ClusteringMethod):
    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        # dbscan dont require the number of clusters it determines it bases
        # on min_samples per cluster and eps (max distance between two samples to be
        # considered in one cluster)
        self.dbscan = cluster.DBSCAN(min_samples)

    def __call__(self, x):
        return self.dbscan.fit_predict(x)

class BGMM(ClusteringMethod):
    def __init__(self, n_clusters: int = 10, weight_concentration_prior: float = 0.1):
        # set n_components if u have a intuiation how many clusters might exists (better starting point and faster convergence)
        # bayesian cause its better for unknown num of clusters
        self.gmm = mixture.BayesianGaussianMixture(n_components=n_clusters, weight_concentration_prior=weight_concentration_prior)

    def __call__(self, x):
        return self.gmm.fit_predict(x)

class SpectralClustering(ClusteringMethod):
    def __init__(self, n_clusters: int):
        self.spectral = cluster.SpectralClustering(n_clusters)

    def __call__(self, x):
        return self.spectral.fit_predict(x)
