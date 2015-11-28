import json
import os
import re
import sys
from scipy import sparse

import numpy as np

from helper import error

__author__ = 'lee'

# the global stop word list
_stopword_set = None

# if is parsing data for liblinear
_svm = False


def _hashcode(token):
    """
    Compute the hashcode of a word.
    The hash function is: h(w) = sum from i = 0 to n - 1(ascii_code(wi)*(31^(n-i))),
    where n is the length of w, wi is the ith character.
    :param token: a word
    :return: the hashcode
    """
    res = 0
    l = len(token)
    cnt = 1
    for c in token:
        res += ord(c) * 31 ** (l - cnt)
        cnt += 1
    return res


def _stopwords():
    """
    Read the stop words file, and return a set of stop words.
    :return: a set of stop words.
    """
    global _stopword_set
    if _stopword_set:
        return _stopword_set
    f_name = "stopword.list"
    if os.path.isfile(f_name):
        res = set()
        with open(f_name) as f:
            for line in f:
                res.add(line.strip())
        _stopword_set = res
        return res
    else:
        error("stop words - not a file: %s" % f_name)


def _extract(txt):
    """
    Extract all legal words given a piece of text
    :param txt: the text
    :return: a word list
    """
    words = []
    blank = re.compile(r"\s+")
    num = re.compile(r"[0-9]")
    punc = re.compile(r"[^a-z]")
    for t in blank.split(txt.lower()):
        # omit words that are stop words or contain numbers
        if num.match(t):
            continue
        # remove punctuations
        t = punc.sub("", t)
        if t == "" or t in _stopwords():
            continue
        words.append(t)
    return words


def _extract_words(f_name):
    """
    Index and count the words in a raw data file.
    :param f_name: the raw data file
    :return: the word:index dict, the index:word dict, and the word count
    """
    word_dict = {}
    idx_dict = {}
    word_cnt = []
    wc = 0
    if os.path.isfile(f_name):
        with open(f_name) as f:
            for line in f:
                obj = json.loads(line)
                # process words in the text
                for t in _extract(obj["text"]):
                    # update the word counts
                    if t not in word_dict:
                        word_dict[t] = wc
                        idx_dict[wc] = t
                        wc += 1
                        word_cnt.append(1)
                    else:
                        word_cnt[word_dict[t]] += 1
        return word_dict, idx_dict, word_cnt
    else:
        error("parse dict - not a file: %s" % f_name)


def _top_n_words(n, f_name):
    """
    Extract the top n words in terms of df from a json file.
    :param n: number of words to return
    :param f_name: the json file
    :return: list of words in reverse order based on the df's
    """
    word_dict, idx_dict, word_cnt = _extract_words(f_name)
    print ("number of words: %d" % len(word_cnt))
    n = min(len(word_cnt), n)
    np_cnt = np.array(word_cnt)
    idx = np.argpartition(np_cnt, -n)[-n:]
    res = []
    for i in idx:
        res.append((idx_dict[i], np_cnt[i]))
    res.sort(key=lambda t: t[1], reverse=True)
    return res


def _explore(f_name):
    """
    Explore the data from the file.
    :param f_name: the raw data file
    :return: None.
    """
    print _top_n_words(10, f_name)


def _baseline_dict(f_name):
    """
    Generate a normal dictionary from a raw data file, and
    write the results to a file.
    :param f_name: the raw data file
    :return: None.
    """
    res = _top_n_words(1000, f_name)
    with open("baseline_dict.txt", "w") as f:
        idx = 0
        for w, c in res:
            f.write("%s %d\n" % (w, idx))
            idx += 1


def _hashing_dict(f_name):
    """
    Generate a hashed dictionary from a raw data file, and
    write the results to a file.
    :param f_name: the raw data file
    :return: None.
    """
    res = _top_n_words(10000, f_name)
    with open("hashing_dict.txt", "w") as f:
        for w, c in res:
            f.write("%s %d\n" % (w, _hashcode(w) % 1000))


def _parse_word_dict(dict_f_name):
    """
    Read in the dictionary file and convert the dictionary
    to a python dict
    :param dict_f_name: dictionary file
    :return: dict of the dictionary { word: index, ... }
    """
    if os.path.isfile(dict_f_name):
        word_dict = {}
        with open(dict_f_name) as f:
            for line in f:
                w, idx = line.strip().split(" ")
                word_dict[w.strip()] = int(idx)
        return word_dict
    else:
        error("Dict not exists: %s" % dict_f_name)


def _parse_json(model, f_name):
    """
    Parse the raw data into its sparse matrix representation format
    using the precomputed dictionary, and write the results to a file.
    :param model: whether baseline or hashing
    :param f_name: the json file to be parsed
    :return: None.
    """
    # get the word index dictionary corresponding to the feature model type
    if model == "baseline":
        word_dict = _parse_word_dict("baseline_dict.txt")
    elif model == "hashing":
        word_dict = _parse_word_dict("hashing_dict.txt")
    elif model == "cluster":
        word_dict = _parse_word_dict("cluster_dict.txt")
    else:
        error("Unknown model type %s" % model)

    if os.path.isfile(f_name):
        if _svm:
            model += "svm"
        out = open("datasets/%s_%s.txt" % (f_name[f_name.rfind("/") + 1:].split(".")[0], model), "w")
        with open(f_name) as f:
            for line in f:
                obj = json.loads(line)
                txt = obj["text"]
                rat = obj["stars"] if "stars" in obj else 0
                out.write("%d \t" % rat)
                features = []
                for t in _extract(txt):
                    if t in word_dict:
                        while len(features) <= word_dict[t]:
                            features.append(0)
                        features[word_dict[t]] += 1
                for i, c in enumerate(features):
                    if c == 0:
                        continue
                    if _svm:
                        i += 1
                    out.write("%d:%d " % (i, c))
                out.write("\n")
        out.close()
    else:
        error("parse json - not a file: %s" % f_name)


def _clz_word_mat(f_name):
    """
    Generate the matrix which is the input to clustering.
    The matrix is 10000 by 5 (number of words, number of ratings).
    The matrix will be written to the file rating_words_condense.txt
    in sparse matrix format.
    :param f_name: the raw json file
    :return: None
    """
    if os.path.isfile(f_name):
        # parse the dictionary containing the top 10000 words
        word_dict = _parse_word_dict("10000word_dict.txt")
        # rating dimension
        x = []
        # word dimension
        y = []
        # occurrence
        data = []
        with open(f_name) as f:
            for line in f:
                obj = json.loads(line)
                txt = obj["text"]
                if "stars" not in obj:
                    raise ValueError("rating doesn't exist.")
                rat = obj["stars"]
                for t in _extract(txt):
                    if t in word_dict:
                        x.append(word_dict[t])
                        y.append(rat - 1)
                        data.append(1)
        A = sparse.coo_matrix((data, (x, y))).tocsr()
        with open("rating_words_condense.txt", "w") as f:
            for i, j in zip(A.nonzero()[0], A.nonzero()[1]):
                # word index, rating, count
                f.write("%d %d %d\n" % (i, j, A[i, j]))
    else:
        error("class word matrix - not a file: %s" % f_name)


def parse_model(f_name):
    """
    Parse the parameters (w's) and return a (5 by m) numpy matrix.
    :param f_name: the file containing the model
    :return: a (5 by m) numpy matrix
    """
    if os.path.isfile(f_name):
        with open(f_name) as f:
            w = [[], [], [], [], []]
            for i, line in enumerate(f):
                for v in line.strip().split(" "):
                    w[i].append(float(v))
            return np.matrix(w)
    else:
        error("parse model - not a file: %s" % f_name)


def parse_matrix(f_name):
    """
    Parse the data (x's and y's) and return them as scipy csr sparse matrices.
    :param f_name: the file containing x vectors and y
    :return: x, y as scipy csr sparse matrices.
    """
    if os.path.isfile(f_name):
        rat_x = []
        rat_y = []
        rat_data = []
        cnt_x = []
        cnt_y = []
        cnt_data = []
        with open(f_name) as f:
            for i, line in enumerate(f):
                rat, vec = line.split("\t")
                if vec.strip() == "":
                    continue
                rat_x.append(i)
                rat_y.append(0 if int(rat) == 0 else int(rat) - 1)
                rat_data.append(1)
                for idx, cnt in [(ic.split(":")) for ic in vec.strip().split(" ")]:
                    cnt_x.append(i)
                    cnt_y.append(idx)
                    cnt_data.append(int(cnt))
        x = sparse.csr_matrix((cnt_data, (cnt_x, cnt_y)))
        y = sparse.csr_matrix((rat_data, (rat_x, rat_y)))
        return x, y
    else:
        error("parse matrix - not a file: %s" % f_name)


def _usage():
    print "Usage: python parser.py [parse] [baseline|hashing] jsonfile.json"

if __name__ == "__main__":
    # generate sparse matrix based on a dictionary
    if len(sys.argv) == 4:
        if sys.argv[1] == "parse":
            _parse_json(sys.argv[2], sys.argv[3])
        else:
            _usage()
    # generate dictionary
    elif len(sys.argv) == 3:
        if sys.argv[1] == "baseline":
            _baseline_dict(sys.argv[2])
        elif sys.argv[1] == "hashing":
            _hashing_dict(sys.argv[2])
        # generate rating-word matrix
        elif sys.argv[1] == "rating-word":
            _clz_word_mat(sys.argv[2])
        else:
            _usage()
    # corpus exploration
    elif len(sys.argv) == 2:
        _explore(sys.argv[1])
    else:
        _usage()
