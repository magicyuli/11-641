import os
from parser import parse_indri

__author__ = 'lee'


class Retriever:
    def __init__(self, pr_type, pr, w_type):
        """
        Do retrieval based on the PR provided and weighting strategy specified.
        :param pr_type: PageRank type
        :param pr: PageRank results
        :param w_type: weighting strategy
        :return: None
        """
        self._pr_type = pr_type.upper()
        self._pr = pr
        self._w_type = w_type.upper()

    def retrieve(self):
        """
        Starting point of retrieval.
        :return: None
        """
        indri_dir = "hw3-resources/indri-lists"
        if not os.path.isdir(indri_dir):
            raise RuntimeError("Cannot find " + indri_dir)
        if not os.path.isdir("retrieval"):
            os.mkdir("retrieval")

        with open("retrieval/%s-%s.results.txt" % (self._pr_type, self._w_type), "w") as out:
            for f in os.listdir(indri_dir):
                indri_id = f.split(".")[0]
                indri = parse_indri(indri_dir + "/" + f)
                if self._pr_type == "GPR":
                    self._retrieve(self._pr, indri_id, indri, out)
                elif self._pr_type == "PTSPR" or self._pr_type == "QTSPR":
                    self._retrieve(self._pr[indri_id], indri_id, indri, out)
                else:
                    raise RuntimeError("PR type unknown.")

    def _retrieve(self, pr, indri_id, indri, out):
        """
        Do the final score calculation and write to the results file.
        :param pr: PageRank scores to use
        :param indri: doc_id, relevance score pairs
        :param indri_id: user-query
        :param out: output file
        :return: None
        """
        row_format = "%s Q0 %d %d %f lee\n"
        scores = []
        # apply different weighting strategy based on the weighting type
        if self._w_type == "NS":
            for doc_id, rel in indri:
                scores.append((doc_id, pr[doc_id]))
        elif self._w_type == "WS":
            for doc_id, rel in indri:
                scores.append((doc_id, pr[doc_id] * 1000 + rel * 0.5))
        elif self._w_type == "CM":
            # get max relevance score
            rel_max = indri[0][1]
            rel_min = indri[len(indri) - 1][1]
            for doc_id, rel in indri:
                scores.append((doc_id, pr[doc_id] * 1000 * (rel_max - rel) / (rel_max - rel_min) + rel * 0.5))
        else:
            raise RuntimeError("weighting strategy unknown.")
        # rank the docs
        scores.sort(key=lambda e: e[1], reverse=True)
        rank = 1
        # output to file
        for doc_id, score in scores:
            out.write(row_format % (indri_id, doc_id, rank, score))
            rank += 1

