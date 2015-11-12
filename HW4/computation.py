import numpy

__author__ = 'lee'


class KNNComputer:
    def __init__(self, mat, model=False):
        """
        Init the knn computer.
        :param mat: x1-x2 matrix
        :param model: if using model-based methods
        """
        self._mat = mat
        self._model = model
        if not self._model:
            # pre-compute L2-norms
            # dense matrix (when model-based)
            if type(self._mat) == numpy.matrixlib.defmatrix.matrix:
                all_norms = numpy.linalg.norm(self._mat, axis=1)
            # sparse matrix (when memory based)
            else:
                all_norms = numpy.linalg.norm(self._mat.toarray(), axis=1)
            all_norms[all_norms == 0] = float("Inf")
            self._all_norms = all_norms

    def knn(self, idx, k, cos=False):
        """
        Find kNN for obj based on mat and similarity metric
        :param idx: index of one vector to find similarity for
        :param k: neighborhood number
        :param cos: whether to use cosine similarity. Effective only when not using model-based
        :return: list of tuples. Every tuple contains (user_id, similarity)
        """
        if self._model:
            if type(self._mat) == numpy.matrixlib.defmatrix.matrix:
                sims = numpy.asarray(self._mat[idx])[0]
            else:
                sims = self._mat[idx].toarray()[0]
        else:
            target = self._mat[idx].toarray()[0]
            # compute dot products
            sims = self._mat.dot(target)
            if cos:
                # divide dot products by norms
                target_norm = self._all_norms[idx]
                sims = sims / self._all_norms / target_norm
        # get top k + 1 to exclude self
        ind = numpy.argpartition(sims, -(k + 1))[-(k + 1):]
        res = []
        for i in ind:
            res.append((i, sims[i]))
        # sort by similarity
        res.sort(key=lambda t: t[1], reverse=True)
        # has negative similarities
        if res[-1][1] < 0:
            self._make_positive(res)
        return res[1:]

    def _make_positive(self, vec):
        """
        Helper function to make similarities positive
        :param vec: similarity vector
        :return: non-negative similarity vector
        """
        new_vec = []
        for e in vec:
            new_vec.append((e[0], e[1] - vec[-1][1]))
