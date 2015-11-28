#!/usr/bin/env bash
awk 'BEGIN{c = 0}{print $1" "c++;}' ./hashing_dict.txt > 10000word_dict.txt