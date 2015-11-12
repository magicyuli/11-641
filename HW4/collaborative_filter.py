import os
from computation import KNNComputer

__author__ = 'lee'


def _run_knn_based(knn_comp, mat, q_file, k, cf_type, weighted_mean=False, cos_sim=False, clustered=False, c_dict=None):
    """
    Run knn based algorithms
    :param knn_comp: the knn computer instance
    :param mat: the user-movie | movie-user | user-cluster-movie | movie-cluster-user matrix
    :param q_file: the query input file
    :param k: k for kNN
    :param cf_type: "user" or "item"
    :param weighted_mean: whether to use weighted mean
    :param cos_sim: whether to use cosine similarity
    :param clustered: whether to use clustered data
    :param c_dict: the data point to cluster dictionary
    :return: a vector of predicted ratings
    """
    if os.path.isfile(q_file):
        # cached knn for users for avoiding repetitive computations
        kn_dict = {}
        predict = []
        with open(q_file) as f:
            for line in f:
                item, user = line.split(",")
                if cf_type.lower() == "user":
                    x1 = int(user)
                    x2 = int(item)
                elif cf_type.lower() == "item":
                    x2 = int(user)
                    x1 = int(item)
                else:
                    raise ValueError("Unknown cf type: %s" % cf_type)
                # knn for x1 hasn't been computed
                if x1 not in kn_dict:
                    if clustered:
                        x1 = c_dict[x1]
                    kn = knn_comp.knn(x1, k, cos_sim)
                    kn_dict[x1] = kn
                # knn for x1 has been computed
                else:
                    kn = kn_dict[x1]

                # calc sum of rating
                pred_rating = 0.0
                sum_sim = 0.0
                # user/item, similarity (non-negative) in knn
                for other_x1, sim in kn:
                    r = mat[other_x1, x2]
                    pred_rating += r if not weighted_mean else r * sim
                    sum_sim += sim
                # weighted mean
                if weighted_mean:
                    if sum_sim > 0.0:
                        pred_rating /= sum_sim
                    else:
                        pred_rating = 0.0
                # normal mean
                else:
                    pred_rating /= len(kn)
                predict.append(pred_rating)
        return predict
    else:
        raise IOError("No such file: %s" % q_file)


def run_memory_based(ui_mat, q_file, k, cf_type, weighted_mean=False, cos_sim=False):
    knn_comp = KNNComputer(ui_mat)
    return _run_knn_based(knn_comp, ui_mat, q_file, k, cf_type, weighted_mean, cos_sim)


def run_model_based(iu_mat, model, q_file, k, cf_type, weighted_mean=False):
    knn_comp = KNNComputer(model, model=True)
    return _run_knn_based(knn_comp, iu_mat, q_file, k, cf_type, weighted_mean)


def run_bipartite_memory_based(ui_mat, q_file, k, cf_type, c_dict, weighted_mean=False, cos_sim=False):
    knn_comp = KNNComputer(ui_mat)
    return _run_knn_based(knn_comp, ui_mat, q_file, k, cf_type, weighted_mean, cos_sim, clustered=True, c_dict=c_dict)


def run_bipartite_model_based(iu_mat, model, q_file, k, cf_type, c_dict, weighted_mean=False):
    knn_comp = KNNComputer(model, model=True)
    return _run_knn_based(knn_comp, iu_mat, q_file, k, cf_type, weighted_mean, clustered=True, c_dict=c_dict)
