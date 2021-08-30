#!/usr/bin/env python3
import re, sys, os
import random

print("Splitting corpus into train/valid/test ...")

corpus = []
with open(sys.argv[1]) as infile:
    for line in infile:
        corpus.append([line.strip()])

with open(sys.argv[2]) as infile:
    for idx,line in enumerate(infile):
        corpus[idx].append(line.strip())

outdir = sys.argv[3]
src = sys.argv[4]
tgt = sys.argv[5]

random.shuffle(corpus)

valid = 0.08 * len(corpus)
test = 10000

tr_s = open(f"{outdir}/train.{src}", "w")
tr_t = open(f"{outdir}/train.{tgt}", "w")

vl_s = open(f"{outdir}/valid.{src}", "w")
vl_t = open(f"{outdir}/valid.{tgt}", "w")

te_s = open(f"{outdir}/test.{src}", "w")
te_t = open(f"{outdir}/test.{tgt}", "w")

for idx, item in enumerate(corpus):
    if idx < test:
        print(item [0], file=te_s)
        print(item [1], file=te_t)
    elif idx < test + valid:
        print(item [0], file=vl_s)
        print(item [1], file=vl_t)
    else :
        print(item [0], file=tr_s)
        print(item [1], file=tr_t)

print(f"Done. Saved files to {outdir}.")
