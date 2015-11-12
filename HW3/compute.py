from collections import defaultdict
import math

__author__ = 'lee'


def scalar_times_vector(n, v):
    res = [0] * len(v)
    for i in range(0, len(v)):
        res[i] = v[i] * n
    return res


def vector_plus_vector(v1, v2):
    if len(v1) != len(v2):
        raise RuntimeError("Lengths of two vectors are not equal.")
    res = [0] * len(v1)
    for i in range(0, len(v1)):
        res[i] = v1[i] + v2[i]
    return res


def distance(v1, v2):
    if len(v1) != len(v2):
        raise RuntimeError("Lengths of two vectors are not equal.")
    ss = 0.0
    for i in range(0, len(v1)):
        ss += (v1[i] - v2[i]) ** 2
    return math.sqrt(ss)


def transpose(m):
    m_t = defaultdict(list)
    for i, row in m.iteritems():
        for j, v in row:
            m_t[j].append((i, v))
    return m_t
