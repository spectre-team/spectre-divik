import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.manifold import SpectralEmbedding
from sklearn.utils.validation import check_is_fitted

from spdivik.distance import DistanceMetric, make_distance
from spdivik.types import Data


def locally_adjusted_affinity(d: DistanceMetric, X: Data, neighbors: int = 7) \
        -> Data:
    """Calculate affinity with local density correction

    Calculate affinity matrix based on input coordinates matrix and the number
    of nearest neighbors.

    Apply local scaling based on the k nearest neighbor

    Parameters
    ----------
    d : DistanceMetric
        Measure of distance between points.

    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster.

    neighbors : int
        The number of neighbors considered a local neighborhood.

    Returns
    -------

    affinity : array, shape [n_samples, n_samples]
        Adjusted affinity matrix.

    References:
    ----------
    https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf

    """
    distances = d(X, X)
    knn_distances = np.sort(distances, axis=0)[neighbors].reshape(-1, 1)
    local_scale = knn_distances.dot(knn_distances.T)
    affinity = - distances ** 2 / local_scale
    affinity[np.isnan(affinity)] = 0
    affinity = np.exp(affinity)
    np.fill_diagonal(affinity, 0)
    return affinity


class LocallyAdjustedRbfSpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Parameters
    -----------
    distance : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
    'kulsinski', 'mahalanobis', 'atching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
        Distance measure, defaults to 'euclidean'. These are the distances
        supported by scipy package.


    n_components : integer, default: 2
        The dimension of the projected subspace.

    random_state : int, RandomState instance or None, optional, default: None
        A pseudo random number generator used for the initialization of the
        lobpcg eigenvectors.  If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``solver`` ==
        'amg'.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    n_neighbors : int, default : max(n_samples/10 , 1)
        Number of nearest neighbors for nearest_neighbors graph building.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------

    embedding_ : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.

    affinity_matrix_ : array, shape = (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    References
    ----------

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - On Spectral Clustering: Analysis and an algorithm, 2001
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
    """
    def __init__(self, distance: str = 'euclidean', n_components=2,
                 random_state=None, eigen_solver: str = None,
                 n_neighbors: int = None, n_jobs: int = 1):
        self.distance = distance
        self.n_components = n_components
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        Y: Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        logging.debug('Computing locally adjusted affinities.')
        distance = make_distance(self.distance)
        affinity_matrix_ = locally_adjusted_affinity(
            distance, X, self.n_neighbors)

        logging.debug('Computing embedding of affinities.')
        embedder = SpectralEmbedding(n_components=self.n_components,
                                     affinity='precomputed',
                                     gamma=None,
                                     random_state=self.random_state,
                                     eigen_solver=self.eigen_solver,
                                     n_neighbors=self.n_neighbors,
                                     n_jobs=self.n_jobs)
        self.embedding_ = embedder.fit_transform(affinity_matrix_)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        Y: Ignored.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return self.fit(X).embedding_

    def save(self, destination: str):
        """Save embedding to a directory

        Parameters
        ----------
        destination : str
            Directory to save the embedding.
        """
        logging.info('Saving embedding to {0}.'.format(destination))
        check_is_fitted(self, 'embedding_')
        from functools import partial
        import os
        import pickle
        import spdivik._scripting as scr
        fname = partial(os.path.join, destination)

        logging.debug('Saving model.')
        with open(fname('model.pkl'), 'wb') as pkl:
            pickle.dump(self, pkl)

        logging.debug('Saving embedding.')
        scr.save_csv(self.embedding_, fname('embedding.csv'))
        np.save(fname('embedding.npy'), self.embedding_)


def main():
    import spdivik._scripting as scr
    data, config, destination, _ = scr.initialize()
    try:
        spectral = LocallyAdjustedRbfSpectralEmbedding(**config)
        spectral.fit(data)
        spectral.save(destination)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise


if __name__ == '__main__':
    main()
