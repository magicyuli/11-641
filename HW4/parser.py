from collections import defaultdict
from scipy import sparse
import os

import numpy as np

__author__ = 'lee'


def parse_training_data(f_name, imputation=True):
    """
    Parse the training data from file
    :param f_name: file name
    :return: u-to-i matrix and i-to-u matrix (both in dictionary format)
    """
    if os.path.isfile(f_name):
        max_uid = 0
        max_iid = 0
        # record count
        r_cnt = 0
        # counts for each rating
        cnt = defaultdict(int)
        # if perform imputation
        if imputation:
            adj = -3
        else:
            adj = 0

        u_idx = []
        i_idx = []
        R = []
        with open(f_name) as f:
            for line in f:
                iid, uid, rating, date = line.split(",")
                uid = int(uid)
                iid = int(iid)
                rating = int(rating) + adj
                u_idx.append(uid)
                i_idx.append(iid)
                R.append(rating)
                cnt[rating] += 1

                max_uid = uid if uid > max_uid else max_uid
                max_iid = iid if iid > max_iid else max_iid
                r_cnt += 1
        I = np.array(u_idx)
        J = np.array(i_idx)
        ui_mat = sparse.coo_matrix((R, (I, J))).tocsr()
        iu_mat = sparse.coo_matrix((R, (J, I))).tocsr()

        return ui_mat, iu_mat
    else:
        raise IOError("No such file: %s" % f_name)


def parse_query_training_data(f_name, imputation=True):
    """
    Parse the training data from the *.queries file
    :param f_name: file name
    :return: u-to-i matrix and i-to-u matrix (both in dictionary format)
    """
    if os.path.isfile(f_name):
        # item by user matrix
        i_u_mat = defaultdict(list)
        # user by item matrix
        u_i_mat = defaultdict(list)
        # if perform imputation
        if imputation:
            adj = -3
        else:
            adj = 0

        with open(f_name) as f:
            for line in f:
                line = line.strip()
                if line.find(" ") < 0:
                    continue
                user, ratings = line.split(" ", 1)
                user = int(user)
                for i_r in ratings.split(" "):
                    item, rating = i_r.split(":")
                    item = int(item)
                    rating = int(rating) + adj
                    u_i_mat[user].append((item, rating))
                    i_u_mat[item].append((user, rating))
        for k, v in u_i_mat.iteritems():
            v.sort(key=lambda i: i[0])

        for k, v in i_u_mat.iteritems():
            v.sort(key=lambda i: i[0])
        return u_i_mat, i_u_mat
    else:
        raise IOError("No such file: %s" % f_name)
