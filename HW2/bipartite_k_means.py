import random
from datetime import datetime
import sys

__author__ = 'lee'


# noinspection PyMethodMayBeStatic
class BipartiteClusterer:
    def __init__(self, dim_x_mat=None, dim_y_mat=None, k_x=None, k_y=None, x_clusters=None, y_clusters=None):
        # matrix of dimension x, e.g, doc
        self._x_mat = dim_x_mat
        # matrix of dimension y, e.g, word
        self._y_mat = dim_y_mat
        # cluster number for dimension x
        self._k_x = k_x
        # cluster number for dimension y
        self._k_y = k_y
        # x_clustering result
        self._x_clusters = x_clusters
        # y_clustering result
        self._y_clusters = y_clusters

    def _cos_sim(self, v1, v2):
        """
        Compute the cosine similarity of two invert-indexed vectors represented by a list of tuples.
        v = [(attribute_3, weight_3), (attribute_8, weight_8), ...].
        Tuples should be sorted by attributes.
        :param v1: vector 1
        :param v2: vector 2
        :return: cosine similarity of v1 and v2
        """
        if not len(v1) or not len(v2):
            return 0

        norm_1 = 0
        norm_2 = 0
        dot_prod = 0
        for k, v in v1:
            norm_1 += v ** 2
        for k, v in v2:
            norm_2 += v ** 2
        i = 0
        j = 0
        while i < len(v1) and j < len(v2):
            if v1[i][0] < v2[j][0]:
                i += 1
            elif v1[i][0] > v2[j][0]:
                j += 1
            else:
                dot_prod += v1[i][1] * v2[j][1]
                i += 1
                j += 1
        return dot_prod / ((norm_1 * norm_2) ** 0.5)

    def _centroid(self, cluster, matrix):
        """
        Compute the centroid of a cluster.
        The cluster contains of list of vectors(lists of tuples).
        Each vector is a invert-indexed item
        :param cluster: the cluster containing item indexes
        :return: an vector
        """
        if len(cluster) == 0:
            return []

        c = matrix[cluster[0]]
        # compute the centroid by ci+1 = (ci * i + v) / (i + 1)
        for k, v_i in enumerate(cluster):
            if k == 0:
                continue
            i = 0
            j = 0
            v = matrix[v_i]
            # new attributes from the vector
            new_c = []
            while i < len(c) and j < len(v):
                if c[i][0] < v[j][0]:
                    new_c.append((c[i][0], c[i][1] * k / (k + 1)))
                    i += 1
                elif c[i][0] > v[j][0]:
                    new_c.append((v[j][0], v[j][1] / (k + 1)))
                    j += 1
                else:
                    new_c.append((c[i][0], (c[i][1] * k + v[j][1]) / (k + 1)))
                    i += 1
                    j += 1
            # remaining attributes in current centroid
            while i < len(c):
                new_c.append((c[i][0], c[i][1] * k / (k + 1)))
                i += 1
            # remaining attributes in the vector
            while j < len(v):
                new_c.append((v[j][0], v[j][1] / (k + 1)))
                j += 1
            c = new_c
        return c

    def _centroids(self, clusters, matrix):
        new_centroids = []
        for i, cluster in enumerate(clusters):
            new_centroids.append(self._centroid(cluster, matrix))
        return new_centroids

    def _cluster(self, centroids, matrix, i_c_dict):
        """
        Allocate data points to clusters centered at centroids,
        and count the number of data points that traveled between clusters.
        :param centroids: centers of clusters
        :param matrix: data points
        :param i_c_dict: data -> cluster dict
        :return: clusters containing data point indexes
        """
        clusters = [[] for c in centroids]
        change_count = 0
        # allocate items to clusters
        for i, item in enumerate(matrix):
            similarity = -1
            cluster = -1
            for j, cent in enumerate(centroids):
                new_sim = self._cos_sim(cent, item)
                if similarity < new_sim:
                    similarity = new_sim
                    cluster = j
            clusters[cluster].append(i)
            if i_c_dict[i] != cluster:
                i_c_dict[i] = cluster
                change_count += 1
        if change_count / float(len(matrix)) > 0.005:
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
        item_cluster_dict = self._item_cluster_dict(clusters)
        aggr_mat = [[] for i in matrix]
        for i, vec in enumerate(matrix):
            # aggregated weights
            aggr = [0 for c in clusters]
            for x, w in vec:
                aggr[item_cluster_dict[x]] += w
            for c, w in enumerate(aggr):
                if w > 0:
                    aggr_mat[i].append((c, w))
        return aggr_mat

    def _item_cluster_dict(self, clusters=None, item_num=None):
        """
        Construct x -> cluster dict
        :param clusters: clusters of item indexes
        :return: item to cluster dict
        """
        item_cluster_dict = []
        if clusters:
            for i, cluster in enumerate(clusters):
                for index in cluster:
                    while len(item_cluster_dict) <= index:
                        item_cluster_dict.append(-1)
                    item_cluster_dict[index] = i
        elif item_num:
            for i in range(0, item_num):
                item_cluster_dict.append(-1)
        return item_cluster_dict

    def _k_means(self, k_x, k_y, matrix_x, matrix_y):
        """
        Run Bipartite Reinforcement K-means Clustering
        :param k_x: number of clusters x-wise
        :param k_y: number of clusters y-wise
        :param matrix_x: data points x-wise
        :param matrix_y: data points y-wise
        :return: x clusters and y clusters
        """
        if k_x > len(matrix_x) or k_y > len(matrix_y):
            raise ValueError("k smaller than item number")

        # initialization
        x_centroids = random.sample(matrix_x, k_x)
        y_centroids = None
        x_clusters = None
        y_clusters = None
        y_aggregated_matrix = matrix_x
        self._stop = False
        # keep track of items that traveling between clusters
        x_c_dict = self._item_cluster_dict(item_num=len(matrix_x))
        y_c_dict = self._item_cluster_dict(item_num=len(matrix_y))
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
            x_clusters = self._cluster(x_centroids, y_aggregated_matrix, x_c_dict)
            # aggregate x-wise
            x_aggregated_matrix = self._aggregate(x_clusters, matrix_y)
            # compute centroids of current y clusters
            if not y_centroids:
                y_centroids = random.sample(x_aggregated_matrix, k_y)
            else:
                y_centroids = self._centroids(y_clusters, x_aggregated_matrix)
            # cluster y-wise
            y_clusters = self._cluster(y_centroids, x_aggregated_matrix, y_c_dict)
            # aggregate y-wise
            y_aggregated_matrix = self._aggregate(y_clusters, matrix_x)
            # compute centroids of current x clusters
            x_centroids = self._centroids(x_clusters, y_aggregated_matrix)

            print "time used for this round: %f" % (datetime.now() - s_time).total_seconds()
            print "round %d end......." % i
            if self._stop:
                can_stop += 1
            else:
                can_stop = 0

        return x_clusters, y_clusters

    def _scs(self, clusters, matrix):
        """
        Compute the Sum of Cosine Similarity.
        :param clusters: clusters of data point indexes
        :param matrix: data points
        :return: scs
        """
        if not clusters:
            return None
        # compute centroids
        centroids = self._centroids(clusters, matrix)
        scs = 0
        for k, c in enumerate(clusters):
            for i in c:
                scs += self._cos_sim(centroids[k], matrix[i])
        return scs

    def run(self):
        self._x_clusters, self._y_clusters = self._k_means(self._k_x, self._k_y, self._x_mat, self._y_mat)
        return self._x_clusters, self._y_clusters

    def analyze(self):
        scs_x = self._scs(self._x_clusters, self._x_mat)
        scs_y = self._scs(self._y_clusters, self._y_mat)
        return scs_x, scs_y
