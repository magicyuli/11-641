from collections import defaultdict
import sys
from datetime import datetime
from compute import vector_plus_vector, scalar_times_vector, distance

__author__ = 'lee'


class PageRanker:
    """
    The class runs the specific PageRank algorithm based on the
    type provided, then provides the result.
    """
    # iteration converging criterion
    _converging_dist = 1e-5

    def __init__(self, type, m_t, num_of_docs, topic_dict=None, topic_count=None, distro=None):
        """
        Constructor.
        :param type: the type of PageRank algorithm to use
        :param m_t: the transposed transition matrix
        :return:
        """
        self._pr_type = type.upper()
        self._m_t = m_t
        self._num_of_docs = num_of_docs
        self._topic_dict = topic_dict
        self._topic_count = topic_count
        self._distro = distro
        # teleportation vector
        self._e = None
        # remedy for preserving the marcov matrix properties
        self._remedy = 0
        # dampening factor
        self.alpha = 0.9
        self.beta = 0.1
        self.gamma = 0.0
        # iteration results
        self._page_rank = None

    def run(self):
        """
        Triggering point of the iterations
        :return: the resulting PageRank vector
        """
        if self._pr_type == "GPR":
            self._run_gpr()
        elif self._pr_type == "QTSPR" or self._pr_type == "PTSPR":
            self._run_xtspr()
        else:
            raise RuntimeError("PR type unsupported!")

        return self._page_rank

    def _run_gpr(self):
        """
        Initialize the teleportation vector to [1/N...], and start the iteration
        :return: None
        """
        self._e = [1.0 / self._num_of_docs] * self._num_of_docs
        self._page_rank = self._run()

    def _run_xtspr(self):
        """
        Initialize the teleportation vector to [1/N_of_topic...], remedy to 1/N, and start the iteration.
        Compute personalized or query based PageRank after iterations.
        :return: None
        """
        if not self._topic_count or not self._topic_dict or not self._distro:
            raise RuntimeError("Please provide topic dict and topic counts")

        # parameters adjustments
        self.alpha = 0.8
        self.beta = 0.15
        self.gamma = 0.05
        self._remedy = 1.0 / self._num_of_docs
        # topic sensitive PageRank. len(tspr) == len(self._topic_count)
        tspr = []
        # both topic and doc indexes start from 1
        for j in range(1, len(self._topic_count) + 1):
            # calc E
            self._e = [0.0] * (self._num_of_docs + 1)
            for i in range(1, self._num_of_docs + 1):
                if j == self._topic_dict[i]:
                    self._e[i] = 1.0 / self._topic_count[j]
            # give remedy
            # run PR for topic j
            tspr.append(self._run())
        # calc personalized or query based PR based on the distributions
        xtspr = defaultdict(list)
        for user_query, distros in self._distro.iteritems():
            # xtspr[user_query] = sigma(p_t*tspr)
            for i, p in enumerate(distros):
                if not len(xtspr[user_query]):
                    xtspr[user_query] = scalar_times_vector(p, tspr[i])
                else:
                    xtspr[user_query] = vector_plus_vector(xtspr[user_query], scalar_times_vector(p, tspr[i]))
        self._page_rank = xtspr

    def _run(self):
        """
        Do power iterations until converging
        :return: the PR vector
        """
        dist = sys.maxint
        pr = [1.0 / self._num_of_docs] * (self._num_of_docs + 1)
        it_num = 0
        start_time = datetime.now()
        while dist > self._converging_dist:
            print "sum of pr: %f" % sum(pr)
            # to preserve pr values for those nodes which don't have in-links
            # same as putting 1s on the diagonal of the transition matrix
            new_pr = [v for v in pr]
            # power iteration
            for i, row in self._m_t.iteritems():
                # accumulator
                tmp = 0
                for j, v in row:
                    tmp += v * pr[j]
                # new pr element
                new_pr[i] = self.alpha * tmp + self.beta * self._e[i] + self.gamma * self._remedy
            # calc the distance
            dist = distance(pr, new_pr)
            it_num += 1
            print "iteration %d finished, PR moved %.20f" % (it_num, dist)
            pr = new_pr
        end_time = datetime.now()
        print "one pr calculated in %d secs" % (end_time - start_time).total_seconds()
        return pr
