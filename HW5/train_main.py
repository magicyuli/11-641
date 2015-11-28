import os
import sys
from RMLR import RMLRClassifier
from helper import error
from parser import parse_matrix

__author__ = 'lee'


def _usage():
    print "Usage: python train_main.py baseline|hashing|cluster"
    exit(1)


def _write_to_file(w, model):
    with open("models/model_%s.txt" % model, "w") as f:
        n, m = w.shape
        for i in range(0, n):
            for j in range(0, m):
                f.write("%f " % w[i, j])
            f.write("\n")


def _main(model):
    """
    Main procedure for training.
    The trained model will be written to files
    :param model: baseline or hashing or cluster
    :return: None.
    """
    train_file = "datasets/yelp_reviews_train_%s.txt" % model
    if not os.path.isfile(train_file):
        error("train main - not a file: %s" % train_file)
    x, y = parse_matrix(train_file)
    rmlr = RMLRClassifier(x, y)
    rmlr.train()
    _write_to_file(rmlr.get_w(), model)


if __name__ == "__main__":
    if not len(sys.argv) == 2:
        _usage()
    _main(sys.argv[1])
