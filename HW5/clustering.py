from scipy import sparse
from sklearn.cluster import KMeans

__author__ = 'lee'


def _cluster():
    """
    Clustering the words, and write the new dictionary to file.
    :return: None
    """
    # rating dimension
    x = []
    # word dimension
    y = []
    # occurrence
    data = []
    with open("rating_words_condense.txt") as f:
        for line in f:
            idx, rat, cnt = line.strip().split(" ")
            x.append(int(idx))
            y.append(int(rat))
            data.append(int(cnt))
    A = sparse.coo_matrix((data, (x, y))).tocsr()

    kmeans = KMeans(n_clusters=2000, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                    precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=3)
    print "started fitting"
    kmeans.fit(A)
    labels = kmeans.labels_
    print "finished fitting"
    print "started writing new dictionary"
    out = open("cluster_dict.txt", "w")
    with open("10000word_dict.txt") as f:
        for line in f:
            word, idx = line.strip().split(" ")
            # substitute the raw indexes with the cluster labels
            out.write("%s %d\n" % (word, labels[int(idx)]))
    print "finished writing new dictionary"


if __name__ == '__main__':
    _cluster()
