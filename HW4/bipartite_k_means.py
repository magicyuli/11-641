import random
from datetime import datetime
from scipy import sparse
import sys
import numpy as np

__author__ = 'lee'


class BipartiteClusterer:
    def __init__(self, mat=None, k_x=None, k_y=None, x_clusters=None, y_clusters=None):
        """
        Init the clusterer
        :param mat: matrix of data (numpy matrix)
        :param k_x: cluster number for dimension x
        :param k_y: cluster number for dimension y
        :param x_clusters: x_clustering result
        :param y_clusters: y_clustering result
        :return:
        """
        self._x_mat = mat
        self._y_mat = mat.T
        self._k_x = k_x
        self._k_y = k_y
        self._x_clusters = x_clusters
        self._y_clusters = y_clusters

    def _centroids(self, clusters, matrix):
        """
        Compute new centroids.
        :param clusters: (numpy array)
        :param matrix: (numpy matrix)
        :return: new centroids
        """
        clus = matrix[clusters == 0]
        new_cent = [clus.sum(0) / float(clus.shape[0])]
        i = 1
        while i <= np.max(clusters):
            # numpy bug
            try:
                clus = matrix[clusters == i]
                new_cent.append(clus.sum(0) / float(clus.shape[0]))
            except ValueError as e:
                new_cent.append(np.zeros((1, matrix.shape[1])))
            i += 1
        return sparse.csr_matrix(np.vstack(tuple(new_cent)))

    def _cluster(self, centroids, matrix, old_clu):
        """
        Allocate data points to clusters centered at centroids,
        and count the number of data points that traveled between clusters.
        :param centroids: centers of clusters (numpy matrix)
        :param matrix: data points (numpy matrix)
        :param old_clu: data -> cluster dict (numpy array)
        :return: cluster indexes (numpy array)
        """
        # allocate items to clusters
        sim_mat = matrix * centroids.T
        clusters = np.asarray(np.argmax(sim_mat.todense(), 1).reshape(1, -1))[0]
        change_count = matrix.shape[0] if old_clu is None else np.sum((clusters - old_clu) != 0)

        if change_count / float(matrix.shape[0]) > 0.005:
            self._stop = False
        print "item traveled clusters: %d; can stop: %d" % (change_count, self._stop)
        return clusters

    def _aggregate(self, clusters, matrix):
        """
        Aggregate matrix row-wisely based on clusters.
        :param clusters:
        :param matrix:
        :return: aggregated matrix
        """
        return self._centroids(clusters, matrix)

    def _k_means(self, k_x, k_y, matrix_x, matrix_y):
        """
        Run Bipartite Reinforcement K-means Clustering
        :param k_x: number of clusters x-wise
        :param k_y: number of clusters y-wise
        :param matrix_x: data points x-wise
        :param matrix_y: data points y-wise
        :return: x clusters and y clusters
        """
        if k_x > matrix_x.shape[0] or k_y > matrix_y.shape[0]:
            raise ValueError("k smaller than item number")

        # initialization
        x_centroids = sparse.csr_matrix(matrix_x[random.sample([i for i in range(matrix_x.shape[0])], k_x)])
        y_centroids = None
        x_clusters = None
        y_clusters = None
        y_aggregated_matrix = matrix_x
        self._stop = False
        can_stop = 0
        i = 0
        while can_stop < 5:
            # converged
            i += 1
            self._stop = True
            print "round %d begin......." % i
            sys.stdout.flush()
            s_time = datetime.now()

            # cluster x-wise
            x_clusters = self._cluster(x_centroids, y_aggregated_matrix, x_clusters)
            # aggregate x-wise
            x_aggregated_matrix = self._aggregate(x_clusters, matrix_x).T
            # compute centroids of current y clusters
            if y_centroids is None:
                y_centroids = sparse.csr_matrix(x_aggregated_matrix[random.sample(
                    [idx for idx in range(x_aggregated_matrix.shape[0])], k_y)])
            else:
                y_centroids = self._centroids(y_clusters, x_aggregated_matrix)
            # cluster y-wise
            y_clusters = self._cluster(y_centroids, x_aggregated_matrix, y_clusters)
            # aggregate y-wise
            y_aggregated_matrix = self._aggregate(y_clusters, matrix_y).T
            # compute centroids of current x clusters
            x_centroids = self._centroids(x_clusters, y_aggregated_matrix)

            print "time used for this round: %f" % (datetime.now() - s_time).total_seconds()
            print "round %d end......." % i
            if self._stop:
                can_stop += 1
            else:
                can_stop = 0

        x_centroids = self._centroids(x_clusters, matrix_x)
        y_centroids = self._centroids(y_clusters, matrix_y)
        self._x_clusters = x_clusters
        self._y_clusters = y_clusters
        self._x_centroids = x_centroids
        self._y_centroids = y_centroids
        return x_centroids, y_centroids, x_clusters, y_clusters

    def _scs(self, clusters, matrix):
        """
        Compute the Sum of Cosine Similarity.
        :param clusters: clusters of data point indexes
        :param matrix: data points
        :return: scs
        """
        if clusters is None:
            return None
        # compute centroids
        centroids = self._centroids(clusters, matrix)

        return (matrix * centroids.T).sum()

    def run(self):
        return self._k_means(self._k_x, self._k_y, self._x_mat, self._y_mat)

    def analyze(self):
        scs_x = self._scs(self._x_clusters, self._x_mat)
        scs_y = self._scs(self._y_clusters, self._y_mat)
        return scs_x, scs_y
