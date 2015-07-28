#!/bin/env python
# clash_check.py
# Analyzes lineMD_RMSD output based on a set of preselected clashes

from __future__ import division, print_function
from argparse import ArgumentParser
from clash_screen import findClashes
from shared import *
from operator import itemgetter

__author__ = 'Charles Yuan'


def main():
    # Process global variables and paths
    parse()
    global WORKDIR
    WORKDIR = os.getcwd()
    global FIRSTPATH, SECONDPATH
    if args.screen1 is not None and not os.path.isabs(args.screen1):
        FIRSTPATH = WORKDIR + "/" + args.screen1
    else:
        FIRSTPATH = args.screen1
    if args.screen2 is not None and not os.path.isabs(args.screen2):
        SECONDPATH = WORKDIR + "/" + args.screen2
    else:
        SECONDPATH = args.screen2

    # Read input into lists representing categories of residue pairs
    # T->N, C->T and C->N are most of interest, print separately
    first_transitory = []
    first_conserved = []
    second_transitory = []
    second_conserved = []
    with open(FIRSTPATH) as first:
        for line in first.readlines():
            resids = (int(line.split()[1]), int(line.split()[2]))
            if line.startswith("C"):
                first_conserved.append(resids)
            else:  # line.startswith("T")
                first_transitory.append(resids)
    with open(SECONDPATH) as second:
        for line in second.readlines():
            resids = (int(line.split()[1]), int(line.split()[2]))
            if line.startswith("C"):
                second_conserved.append(resids)
            else:  # line.startswith("T")
                second_transitory.append(resids)

    # Compare the before and after lists to yield the differences
    TtoN = []
    CtoT = []
    CtoN = []
    for resPair in first_transitory:
        if (resPair not in second_conserved) and (resPair not in second_transitory):
            # transitory is now nonexistent
            TtoN.append(resPair)
    for resPair in first_conserved:
        if resPair in second_transitory:
            # conserved is now transitory
            CtoT.append(resPair)
        elif resPair not in second_conserved:
            # conserved is now nonexistent
            CtoN.append(resPair)

    # Print the differences
    for resPair in TtoN:
        sys.stdout.write("TN %i %i\n" % (resPair[0], resPair[1]))
    sys.stdout.flush()

    for resPair in CtoT:
        sys.stdout.write("CT %i %i\n" % (resPair[0], resPair[1]))
    sys.stdout.flush()

    for resPair in CtoN:
        sys.stdout.write("CN %i %i\n" % (resPair[0], resPair[1]))
    sys.stdout.flush()


def parse():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Analyze lineMD_RMSD output to "
                                        "compute changed clash residues")
    parser.add_argument("screen1", help="Output file from clash_screen before barrier",
                        type=str, action=FullPath)
    parser.add_argument("screen2", help="Second output file from clash_screen after barrier",
                        type=str, action=FullPath)
    global args
    args = parser.parse_args()


if __name__ == "__main__":
    main()
