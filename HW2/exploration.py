__author__ = 'lee'


def count(docs_f, words_f):
    res = {
        "tot_docs": 0,
        "tot_words": 0,
        "tot_unique_words": 0,
        "avg_unique_words": 0
    }
    unique_word_count = 0
    for line in docs_f:
        res["tot_docs"] += 1
        for w, f in [(pair.split(":")) for pair in line.strip().split(" ")]:
            res["tot_words"] += int(f)
            unique_word_count += 1
    res["avg_unique_words"] = unique_word_count / (res["tot_docs"] * 1.0)
    res["tot_unique_words"] = len(words_f.readlines())
    return res

if __name__ == "__main__":
    dev_docs_f = open("HW2_dev.docVectors", "r")
    dev_words_f = open("HW2_dev.dict", "r")
    res_1 = count(dev_docs_f, dev_words_f)
    dev_docs_f.close()
    dev_words_f.close()

    test_docs_f = open("HW2_test.docVectors", "r")
    test_words_f = open("HW2_test.dict", "r")
    res_2 = count(test_docs_f, test_words_f)
    test_docs_f.close()
    test_words_f.close()

    dev_docs_f = open("HW2_dev.docVectors", "r")
    line_1 = dev_docs_f.readline()
    res_3 = {
        "tot_unique_words": 0,
        "ids": []
    }
    for k, v in [(kv.split(":")) for kv in line_1.strip().split(" ")]:
        res_3["tot_unique_words"] += 1
        if int(v) == 2:
            res_3["ids"].append(int(k))
    dev_docs_f.close()

    print res_1
    print res_2
    print res_3


