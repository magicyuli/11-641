from collections import defaultdict
import os

__author__ = 'lee'


def _check_existence(filename):
    """
    Check if the file pointed by filename exists.
    :param filename: the file
    :return: None
    """
    if not os.path.isfile(filename):
        raise RuntimeError("%s doesn't exist." % filename)


def parse_m(filename):
    """
    Parse the transposed transition matrix from file
    :param filename: the text file.
    :return: the transposed transition matrix as dictionary of list: m_t[n2][n1] exists only if n1 points to n2,
    and the total number of documents.
    """
    _check_existence(filename)
    m = defaultdict(list)
    doc_count = 0
    doc_flag = defaultdict(bool)
    with open(filename) as f:
        for line in f:
            n1, n2, w = (line.strip().split(" "))
            n1 = int(n1)
            n2 = int(n2)
            m[n1].append(n2)
            # count the docs
            if not doc_flag[n2]:
                doc_count += 1
                doc_flag[n2] = True
            if not doc_flag[n1]:
                doc_count += 1
                doc_flag[n1] = True
    for i, row in m.iteritems():
        row.sort()
        for j, v in enumerate(row):
            row[j] = (v, 1.0 / len(row))
    return m, doc_count


def parse_doc_topics(filename):
    """
    Parse the doc topic information from file
    :param filename: the text file
    :return: dictionary in the form of {doc: topic}, and the document counts of each topic
    """
    _check_existence(filename)
    d_t = {}
    counts = []
    with open(filename) as f:
        for line in f:
            d, t = (line.strip().split(" "))
            t = int(t)
            d_t[int(d)] = t
            while len(counts) <= t:
                counts.append(0)
            counts[t] += 1
    return d_t, counts


def parse_distro(filename):
    """
    Parse the personalization/query distribution information from file
    :param filename: the text file
    :return: a dictionary in the form of {user-query: [p1, p2, ...]}
    """
    _check_existence(filename)
    distro_dict = defaultdict(list)
    with open(filename) as f:
        for line in f:
            # split one line into 3 parts
            user, query, distros = (line.strip().split(" ", 2))
            for topic_prob in distros.split(" "):
                # add every probability of topic to the dictionary
                distro_dict["%s-%s" % (user, query)].append(float(topic_prob.split(":")[1]))
    return distro_dict


def parse_indri(filename):
    """
    Parse the indri doc search-relevance scores from the a indri file
    :param filename: the indri file
    :return: list of (doc, score) tuples
    """
    _check_existence(filename)
    res_list = []
    with open(filename) as f:
        for line in f:
            doc = int(line.strip().split(" ")[2])
            score = float(line.strip().split(" ")[4])
            res_list.append((doc, score))
    return res_list


def _parse_pr(filename):
    res = []
    with open(filename) as f:
        for line in f:
            res.append(float(line.strip().split(" ")[1]))
    return res


def parse_pr(pr_type, pr_dir):
    if not os.path.isdir(pr_dir):
        raise RuntimeError("Please provide a directory that contains te PageRank results.")
    if pr_type == "GPR":
        return _parse_pr(os.path.abspath(pr_dir) + "/GPR.txt")
    pr = {}
    for f in os.listdir(pr_dir):
        if f.startswith(pr_type):
            user_query = f.split("-", 1)[1].split(".")[0]
            pr[user_query] = _parse_pr(os.path.abspath(pr_dir) + "/" + f)
    return pr
