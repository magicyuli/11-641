import math

__author__ = 'lee'


def parse_clusters(filename):
    """
    Generate cluster from the cluster files written by clustering jobs
    :param filename: the file name
    :return: the clusters
    """
    clusters_file = open(filename)
    clusters = []
    for line in clusters_file:
        item = int(line.strip().split(" ")[0])
        c = int(line.strip().split(" ")[1])
        while len(clusters) <= c:
            clusters.append([])
        clusters[c].append(item)
    return clusters


def parse_doc_vectors(filename, is_custom=False, df_file=None):
    """
    Construct document matrix and word matrix at the same time from .docVectors file.
    Compute tf-idf if is custom.
    :param filename: .docVectors file
    :param is_custom:
    :param df_file: .df file
    :return: doc matrix and word matrix
    """
    doc_count = 942
    dfs = []
    if is_custom:
        df_f = open(df_file)
        for line in df_f:
            df = float(line.split(":")[1])
            dfs.append(math.log(doc_count / df + 1))
    doc_vec_file = open(filename, "r")
    docs = []
    words = []
    for i, line in enumerate(doc_vec_file):
        docs.append([])
        for word, weight in [(item.split(":")) for item in line.strip().split(" ")]:
            word = int(word)
            weight = float(weight) * dfs[word] if is_custom else float(weight)
            docs[i].append((word, weight))
            while len(words) <= word:
                words.append([])
            words[word].append((i, weight))
        docs[i].sort(key=lambda t: t[0])
    doc_vec_file.close()
    return docs, words
