import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster


class HierarchicalClustering:
    """
    Hierarchical clustering using average linkage starting from
    a correlation matrix.
    """

    def __init__(self, R, verbose=1):
        """
        Parameters
        ----------
        R : ndarray (N x N)
            Correlation matrix with ones on the diagonal.
        verbose : int
            If >= 1, dendrograms are plotted.
            
        """
        self.R = R
        self.verbose = verbose
        
        D = np.sqrt(0.5 * (1 - self.R))
        self.D = D
        
        np.fill_diagonal(self.D, 0.0)
        condensed = squareform(self.D, checks=False)
        self.Z = linkage(condensed, method="average")
        
    def get_dendrogram(self):
        """
        Builds the hierarchical clustering linkage matrix using
        average linkage and optionally plots the dendrogram.

        The correlation-to-distance transformation used is:
            D_ij = sqrt(0.5 * (1 - R_ij))
        """
        if self.verbose >= 1:
            import matplotlib.pyplot as plt
            dendrogram(self.Z)
            plt.show()

    def get_clusters(self, K):
        """
        Cuts the dendrogram to obtain a fixed number of clusters.

        Parameters
        ----------
        K : int
            Number of desired clusters.

        Returns
        -------
        clusters : ndarray
            Cluster label (1..K) assigned to each observation.
        """
        clusters = fcluster(self.Z, t=K, criterion="maxclust")

        return clusters
    
    def choose_k_silhouette(self, k_min=2, k_max=30):
        from sklearn.metrics import silhouette_score
        
        best_ss = - np.inf
        best_k = None
        for k in range(k_min, k_max+1):
            clusters = fcluster(self.Z, t=k, criterion='maxclust')
            ss = silhouette_score(self.D, clusters)
            if ss > best_ss:
                best_k = k
        
        print('best k:', best_k)
        return fcluster(self.Z, t=best_k, criterion='maxclust')
    

