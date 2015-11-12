import os
import sys
from bipartite_k_means import BipartiteClusterer
from parser import parse_clusters, parse_doc_vectors

__author__ = 'lee'

if __name__ == "__main__":
    is_custom = False
    is_word = False
    if len(sys.argv) > 1:
        is_custom = sys.argv[1] == "custom"
        is_word = sys.argv[1] == "word"

    if is_word:
        if len(sys.argv) < 3:
            raise ValueError("please provide word cluster file name")
        f = sys.argv[2]
        if not os.path.isfile(f):
            raise ValueError("please provide correct word cluster file name")
        word_cluster = parse_clusters(f)
        word_dict = []
        w_d_file = open("HW2_dev.dict")
        for line in w_d_file:
            t = (line.strip().split(" "))
            word = t[0]
            index = int(t[1])
            while len(word_dict) <= index:
                word_dict.append("")
            word_dict[index] = word
        for c in word_cluster:
            for i in c:
                sys.stdout.write(word_dict[i] + ", ")
            sys.stdout.write("\n")
    else:
        dev_vecs = "HW2_dev.docVectors"
        dev_df = "HW2_dev.df"

        dev_docs, dev_words = parse_doc_vectors(dev_vecs, is_custom, dev_df)

        output_dir = "outs/"
        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                if (is_custom and not f.startswith("custom_")) or \
                        (not is_custom and f.startswith("custom_")):
                    continue

                clusters = parse_clusters(output_dir + f)
                if f.endswith("_doc"):
                    rss_x, rss_y = BipartiteClusterer(dim_x_mat=dev_docs, x_clusters=clusters).analyze()
                else:
                    rss_x, rss_y = BipartiteClusterer(dim_x_mat=dev_words, x_clusters=clusters).analyze()
                print f
                print rss_x
                sys.stdout.flush()
                if f.endswith("doc"):
                    os.system("python eval.py %s HW2_dev.gold_standards" % (output_dir + f))
                print ""
