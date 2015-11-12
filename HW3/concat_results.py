import os
import time
import sys

__author__ = 'lee'


def concat(dir):
    if not os.path.isdir(dir):
        raise RuntimeError(dir + " is not a directory!")
    with open("result_%d.txt" % int(time.time()), "w") as out:
        for f_name in os.listdir(dir):
            part = f_name.split(".")[0]
            with open(os.path.abspath(dir) + "/" + f_name) as f:
                for line in f:
                    out.write("%s %s" % (part, line.split(" ", 1)[1]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python concat_results.py directory"
        sys.exit(0)
    concat(sys.argv[1])
