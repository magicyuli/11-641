import sys
from datetime import datetime

from compute import transpose
from pagerank import PageRanker
from parser import *
from retrieval import Retriever

__author__ = 'lee'


def _bad_input():
    print "usage: python main.py [NO] GPR|PTSPR|QTSPR [NS|WS|CM]"
    sys.exit(0)


def main():
    if len(sys.argv) < 2:
        _bad_input()

    pr_type = sys.argv[1].upper()

    pr = None
    if not pr_type == "NO":
        m_file = "hw3-resources/transition.txt"
        doc_topic_file = "hw3-resources/doc-topics.txt"
        user_distro_file = "hw3-resources/user-topic-distro.txt"
        query_distro_file = "hw3-resources/query-topic-distro.txt"

        # parse personalization distributions
        distro = None
        doc_topic_dict, doc_topic_count = None, None
        parsing_s_time = datetime.now()
        if pr_type == "GPR":
            pass
        elif pr_type == "PTSPR":
            distro = parse_distro(user_distro_file)
            # parse doc topic info
            doc_topic_dict, doc_topic_count = parse_doc_topics(doc_topic_file)
        elif pr_type == "QTSPR":
            distro = parse_distro(query_distro_file)
            # parse doc topic info
            doc_topic_dict, doc_topic_count = parse_doc_topics(doc_topic_file)
        else:
            _bad_input()

        # parse transition matrix
        m, doc_count = parse_m(m_file)
        # compute the transposed m
        m_t = transpose(m)

        parsing_e_time = datetime.now()
        print "parsing finished in %d secs" % (parsing_e_time - parsing_s_time).total_seconds()

        # init the PageRanker
        pageranker = PageRanker(pr_type, m_t, doc_count, doc_topic_dict, doc_topic_count, distro)
        # run the computations
        pr = pageranker.run()

        pr_e_time = datetime.now()
        print "PageRanking finished in %d secs" % (pr_e_time - parsing_e_time).total_seconds()

        # write to file
        if not os.path.isdir("pageranks"):
            os.mkdir("pageranks")
        if pr_type == "GPR":
            print "sum of pr: %f" % sum(pr)
            with open("pageranks/%s.txt" % pr_type, "w") as f:
                for i, v in enumerate(pr):
                    f.write("%d %.50f\n" % (i, v))
        else:
            # pr contains multiple PageRanks for each user-query
            for user_query, xtspr in pr.iteritems():
                with open("pageranks/%s-%s.txt" % (pr_type, user_query), "w") as f:
                    for i, v in enumerate(xtspr):
                        f.write("%d %.50f\n" % (i, v))

        print "Saving PageRanks to file(s) complete."
    else:
        print "No PageRank will be run."
        # run retrieval only
        if len(sys.argv) >= 3:
            pr_type = sys.argv[2]
            # delete argv[2] so that argv[2] can be weighting strategy
            del sys.argv[2]

    # check if retrieval is required
    if len(sys.argv) < 3:
        print "No retrieval required."
    else:
        # the user chose not to run PageRank
        if not pr:
            pr = parse_pr(pr_type, "pageranks")
        if not pr:
            print "No PageRank records found. Exiting..."
            sys.exit(0)
        retriever = Retriever(pr_type, pr, sys.argv[2])
        print "Starting retrieval..."
        r_s_time = datetime.now()
        retriever.retrieve()
        r_e_time = datetime.now()
        print "Retrieval finished in %d secs." % (r_e_time - r_s_time).total_seconds()

if __name__ == "__main__":
    main()
