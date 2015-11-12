from scipy import sparse
import sys
import time

import numpy as np

from bipartite_k_means import BipartiteClusterer
from collaborative_filter import run_memory_based, run_model_based, run_bipartite_memory_based, \
    run_bipartite_model_based
from parser import parse_training_data

__author__ = 'lee'


def _exit():
    print "Usage: python main.py exp1|exp2|exp3|exp4"
    sys.exit(0)


def _write_to_file(out_f, res, imputated=True):
    adj = 3 if imputated else 0
    with open(out_f, "w") as f:
        for r in res:
            f.write("%f\n" % (r + adj))


def main():
    train_data = "HW4_data/train.csv"
    dev_data = "HW4_data/dev.csv"
    test_data = "HW4_data/test.csv"
    if len(sys.argv) < 2:
        _exit()
    exp_type = sys.argv[1]
    if exp_type == "exp1":
        _run_exp1(dev_data, train_data)
    elif exp_type == "exp2":
        _run_exp2(dev_data, train_data)
    elif exp_type == "exp3":
        _run_exp3(dev_data, train_data)
    elif exp_type == "exp4":
        _run_exp4(dev_data, train_data)
    elif exp_type == "test":
        _run_test(test_data, train_data)
    else:
        _exit()


def _run_test(test_data, train_data):
    ui_mat, iu_mat = parse_training_data(train_data)
    print "################## test k=10 dotp avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, test_data, 10, "user")
    _write_to_file("predictions.txt", res)
    print time.time() - s_t


def _run_exp4(dev_data, train_data):
    k_user = 1000
    k_item = 600
    ui_mat, iu_mat = parse_training_data(train_data)
    clusterer = BipartiteClusterer(ui_mat, k_user, k_item)
    sum_sim = -float("Inf")
    # use the cluster that has the biggest sum of similarity
    for i in range(5):
        u, i, u_c, i_c = clusterer.run()
        s_s = sum(clusterer.analyze())
        if sum_sim < s_s:
            cent_u = u
            cent_i = i
            u_c_dict = u_c
            i_c_dict = i_c
            sum_sim = s_s
    # use centroids
    ui_mat = cent_u
    iu_mat = cent_i

    print "################## k=10 dotp avg ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 10, "user", u_c_dict)
    _write_to_file("exp4_memory_results/10_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 dotp avg ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 100, "user", u_c_dict)
    _write_to_file("exp4_memory_results/100_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 dotp avg ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 500, "user", u_c_dict)
    _write_to_file("exp4_memory_results/500_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos avg ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 10, "user", u_c_dict, cos_sim=True)
    _write_to_file("exp4_memory_results/10_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 cos avg ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 100, "user", u_c_dict, cos_sim=True)
    _write_to_file("exp4_memory_results/100_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 cos avg ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 500, "user", u_c_dict, cos_sim=True)
    _write_to_file("exp4_memory_results/500_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos wei ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 10, "user", u_c_dict, cos_sim=True, weighted_mean=True)
    _write_to_file("exp4_memory_results/10_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=100 cos wei ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 100, "user", u_c_dict, cos_sim=True, weighted_mean=True)
    _write_to_file("exp4_memory_results/100_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=500 cos wei ##################"
    s_t = time.time()
    res = run_bipartite_memory_based(ui_mat, dev_data, 500, "user", u_c_dict, cos_sim=True, weighted_mean=True)
    _write_to_file("exp4_memory_results/500_cos_wei.txt", res)
    print time.time() - s_t

    ##################################################################################################################

    # calculate models
    model = iu_mat.dot(iu_mat.T)
    all_norm = np.linalg.norm(iu_mat.toarray(), axis=1)
    all_norm[all_norm == 0.0] = float("Inf")
    inv_all_norm = (1 / all_norm).reshape(1, -1)
    inv_norm_mat = inv_all_norm.T.dot(inv_all_norm)
    cos_model = sparse.csr_matrix(model.multiply(inv_norm_mat))

    print "################## k=10 dotp avg ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, model, dev_data, 10, "item", i_c_dict)
    _write_to_file("exp4_model_results/10_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 dotp avg ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, model, dev_data, 100, "item", i_c_dict)
    _write_to_file("exp4_model_results/100_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 dotp avg ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, model, dev_data, 500, "item", i_c_dict)
    _write_to_file("exp4_model_results/500_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos avg ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, cos_model, dev_data, 10, "item", i_c_dict)
    _write_to_file("exp4_model_results/10_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 cos avg ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, cos_model, dev_data, 100, "item", i_c_dict)
    _write_to_file("exp4_model_results/100_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 cos avg ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, cos_model, dev_data, 500, "item", i_c_dict)
    _write_to_file("exp4_model_results/500_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos wei ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, cos_model, dev_data, 10, "item", i_c_dict, weighted_mean=True)
    _write_to_file("exp4_model_results/10_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=100 cos wei ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, cos_model, dev_data, 100, "item", i_c_dict, weighted_mean=True)
    _write_to_file("exp4_model_results/100_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=500 cos wei ##################"
    s_t = time.time()
    res = run_bipartite_model_based(iu_mat, cos_model, dev_data, 500, "item", i_c_dict, weighted_mean=True)
    _write_to_file("exp4_model_results/500_cos_wei.txt", res)
    print time.time() - s_t


def _run_exp3(dev_data, train_data):
    ui_mat, iu_mat = parse_training_data(train_data)
    # mean of each row
    mean = iu_mat.mean(1)
    # mean matrix: every row element are the same
    mean_mat = mean * np.ones(iu_mat.shape[1])
    # centered matrix
    centered_mat = iu_mat - mean_mat
    # standard deviation
    stddev = np.sqrt(np.power(centered_mat, 2).sum(1))
    stddev[stddev == 0.0] = float("Inf")
    # inverse standard deviation
    inv_stddev = 1 / stddev
    # inverse standard deviation matrix
    inv_stddev_mat = inv_stddev * np.ones(iu_mat.shape[1])
    # row-normalized matrix
    norm_mat = np.multiply(centered_mat, inv_stddev_mat)
    # offline computed model
    model = norm_mat * norm_mat.T

    print "################## k=10 dotp avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 10, "item")
    _write_to_file("exp3_results/10_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 dotp avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 100, "item")
    _write_to_file("exp3_results/100_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 dotp avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 500, "item")
    _write_to_file("exp3_results/500_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos wei ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 10, "item", weighted_mean=True)
    _write_to_file("exp3_results/10_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=100 cos wei ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 100, "item", weighted_mean=True)
    _write_to_file("exp3_results/100_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=500 cos wei ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 500, "item", weighted_mean=True)
    _write_to_file("exp3_results/500_cos_wei.txt", res)
    print time.time() - s_t


def _run_exp2(dev_data, train_data):
    ui_mat, iu_mat = parse_training_data(train_data)
    model = iu_mat.dot(iu_mat.T)
    all_norm = np.linalg.norm(iu_mat.toarray(), axis=1)
    all_norm[all_norm == 0.0] = float("Inf")
    inv_all_norm = (1 / all_norm).reshape(1, -1)
    inv_norm_mat = inv_all_norm.T.dot(inv_all_norm)
    cos_model = sparse.csr_matrix(model.multiply(inv_norm_mat))

    print "################## k=10 dotp avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 10, "item")
    _write_to_file("exp2_results/10_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 dotp avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 100, "item")
    _write_to_file("exp2_results/100_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 dotp avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, model, dev_data, 500, "item")
    _write_to_file("exp2_results/500_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, cos_model, dev_data, 10, "item")
    _write_to_file("exp2_results/10_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 cos avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, cos_model, dev_data, 100, "item")
    _write_to_file("exp2_results/100_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 cos avg ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, cos_model, dev_data, 500, "item")
    _write_to_file("exp2_results/500_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos wei ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, cos_model, dev_data, 10, "item", weighted_mean=True)
    _write_to_file("exp2_results/10_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=100 cos wei ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, cos_model, dev_data, 100, "item", weighted_mean=True)
    _write_to_file("exp2_results/100_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=500 cos wei ##################"
    s_t = time.time()
    res = run_model_based(iu_mat, cos_model, dev_data, 500, "item", weighted_mean=True)
    _write_to_file("exp2_results/500_cos_wei.txt", res)
    print time.time() - s_t


def _run_exp1(dev_data, train_data):
    ui_mat, iu_mat = parse_training_data(train_data)
    print "################## k=10 dotp avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 10, "user")
    _write_to_file("exp1_results/10_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 dotp avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 100, "user")
    _write_to_file("exp1_results/100_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 dotp avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 500, "user")
    _write_to_file("exp1_results/500_dot_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 10, "user", cos_sim=True)
    _write_to_file("exp1_results/10_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=100 cos avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 100, "user", cos_sim=True)
    _write_to_file("exp1_results/100_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=500 cos avg ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 500, "user", cos_sim=True)
    _write_to_file("exp1_results/500_cos_avg.txt", res)
    print time.time() - s_t

    print "################## k=10 cos wei ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 10, "user", cos_sim=True, weighted_mean=True)
    _write_to_file("exp1_results/10_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=100 cos wei ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 100, "user", cos_sim=True, weighted_mean=True)
    _write_to_file("exp1_results/100_cos_wei.txt", res)
    print time.time() - s_t

    print "################## k=500 cos wei ##################"
    s_t = time.time()
    res = run_memory_based(ui_mat, dev_data, 500, "user", cos_sim=True, weighted_mean=True)
    _write_to_file("exp1_results/500_cos_wei.txt", res)
    print time.time() - s_t


if __name__ == "__main__":
    main()
