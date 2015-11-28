import os
import sys
import numpy as np

from RMLR import RMLRClassifier
from helper import error
from parser import parse_matrix, parse_model

__author__ = 'lee'


def _usage():
    print "Usage: python predict_main.py baseline|hashing train|dev|test"
    exit(1)


def _write_to_file(model, dataset, hard, soft):
    """
    Write the predictions to a file.
    :param model: the name of the model used
    :param dataset: train or dev or test
    :param hard: the hard predictions
    :param soft: the soft predictions
    :return: None
    """
    with open("predictions/prediction_%s_%s.txt" % (dataset, model), "w") as f:
        for h, s in zip(hard, soft):
            # the predictions are on a 0-4 scale
            f.write("%d %f\n" % (h + 1, s + 1))


def _main(model, dataset):
    """
    The main procedure for making predictions
    :param model: the name of the model used
    :param dataset: train or dev or test
    :return: None
    """
    # check the parameters passed
    if model not in ["baseline", "hashing", "cluster"] \
            or dataset not in ["train", "dev", "test"]:
        _usage()
    # the model file
    model_file = "models/model_%s.txt" % model
    # the data file
    data_file = "datasets/yelp_reviews_%s_%s.txt" % (dataset, model)
    if not os.path.isfile(data_file):
        error("predict main - not a file: %s" % dataset)

    # parse the data
    x, y = parse_matrix(data_file)
    # parse the model
    w = parse_model(model_file)
    # init the classifier
    rmlr = RMLRClassifier(w=w)
    # make predictions
    hard, soft = rmlr.predict(x)
    # write the results to file
    _write_to_file(model, dataset, hard, soft)
    # if testing on the training set, report the evaluation results
    if dataset == "train":
        y = np.asarray(np.argmax(y.todense(), 1))
        accu = np.sum(y == hard) / float(y.shape[0])
        rmse = np.sqrt((y - soft).T * (y - soft) / y.shape[0])
        print accu, rmse


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        _usage()
    _main(sys.argv[1], sys.argv[2])
