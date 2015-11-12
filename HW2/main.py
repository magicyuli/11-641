import os
from datetime import datetime
import sys

from parser import parse_doc_vectors
from bipartite_k_means import BipartiteClusterer

__author__ = 'lee'

DOC_K = 98
WORD_K = 84


def write_clusters(clusters, filename):
    out = open(filename, "w")
    items = []
    # collapse the clusters for output
    for i, c in enumerate(clusters):
        for item in c:
            while len(items) <= item:
                items.append(-1)
            items[item] = i
    # write to output
    for d, c in enumerate(items):
        out.write("%d %d\n" % (d, c))
    out.close()

if __name__ == "__main__":
    is_custom = False
    is_test = False
    if len(sys.argv) > 1:
        is_custom = sys.argv[1] == "custom" or sys.argv[1] == "test"
        is_test = sys.argv[1] == "test"

    print "proceed with custom: %s" % str(is_custom)
    print "proceed with test: %s" % str(is_test)

    dev_vecs = "HW2_dev.docVectors"
    dev_df = "HW2_dev.df"
    test_vecs = "HW2_test.docVectors"
    test_df = "HW2_test.df"

    # only run once for test
    if is_test:
        docs, words = parse_doc_vectors(test_vecs, is_custom, test_df)
        num_experiment = 1
    else:
        docs, words = parse_doc_vectors(dev_vecs, is_custom, dev_df)
        num_experiment = 10

    for i in range(0, num_experiment):
        print "###################  experiment %d  #########################" % i

        # clustering
        clusterer = BipartiteClusterer(docs, words, DOC_K, WORD_K)
        s_t = datetime.now()
        doc_clusters, word_clusters = clusterer.run()
        e_t = datetime.now()
        print "time used for clustering: %f" % (e_t - s_t).total_seconds()

        # analyzing
        scs_doc, scs_word = clusterer.analyze()
        print "SCS for document clusters: %f" % scs_doc
        print "SCS for word clusters: %f" % scs_word

        # write results to files
        if is_test:
            job_type = "test"
        elif is_custom:
            job_type = "custom"
        else:
            job_type = "dev"
        doc_out = "outs/%s_%d_doc" % (job_type, i)
        word_out = "outs/%s_%d_word" % (job_type, i)
        write_clusters(doc_clusters, doc_out)
        write_clusters(word_clusters, word_out)

        # evaluate Micro F1
        if not is_test:
            print os.system("python eval.py %s HW2_dev.gold_standards" % doc_out)
        print "###################  end of experiment %d  ######################" % i
