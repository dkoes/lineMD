#!/bin/env python
# clash_screen.py
# Analyzes lineMD_RMSD output to compute changed clash residues

from __future__ import division, print_function
from argparse import ArgumentParser
from itertools import combinations
from shared import *
from numpy import array, linalg
from operator import itemgetter


__author__ = 'Charles Yuan'


def findCombinations(pdbLines):
    """Given a PDB file, calculate possible pairs to check for collisions.
    Also creates the residues database that contains read residues and their atoms.
    Specifies that residues within a certain neighborhood will not be checked."""

    # global database of residues and constituent atoms to be used throughout
    global residues
    residues = {}  # {residueID: [(atomID, atomCoords)]}
    for line in pdbLines:
        # Read line and extract the ID for valid residues
        if line[0:4] != "ATOM" or line[17:20] not in RESIDUES:
            continue
        vals = line.split()
        residueID = int(vals[4])
        if residueID not in residues or residues[residueID] is None:
            # residue has not been previously seen
            residues[residueID] = []
        atomID = int(vals[1])
        atomCoords = [float(line[(30 + i * 8):(38 + i * 8)]) for i in range(3)]
        residues[residueID].append((atomID, atomCoords))
    # calculate possible combinations of the residue IDs
    pairs = [pair for pair in list(combinations(residues.keys(), 2))
             if abs(pair[0] - pair[1]) > 4]  # not within 4 of each other
    return pairs


def findClashes(pdbLines, threshold):
    """Given a PDB and a threshold distance, produce a list of residue pairs
    deemed to be in collision"""
    clashes = set()
    pairs = findCombinations(pdbLines)

    totalChecks = len(pairs)
    log("Total of %i residues, %i possible combinations\n" % (len(residues.keys()), totalChecks))

    def testPair((res1, res2)):
        """Returns whether the residues are in collision"""
        for atom1 in residues[res1]:
            coord1 = atom1[1]
            for atom2 in residues[res2]:
                coord2 = atom2[1]
                dist = linalg.norm(abs(array(coord1) - array(coord2)))
                if dist < threshold:
                    return True
        return False

    # Run testPairs for each combination of residues in parallel
    parallel = cpu_count() / 2  # half number of cores
    output = parMap(testPair, pairs, n=parallel)

    # Print output
    for pairID, out in enumerate(output):
        if out:
            clashes.add((min(pairs[pairID]), max(pairs[pairID])))  # (res1, res2)

    return clashes


def selectFrames(frames, minDist, maxDist, freq):
    """Select frames from frames that are between min and max distance, spliced every freq frames."""
    # find the starting point
    start = 0
    for ID, frameDist in frames:
        if minDist < frameDist < maxDist:
            start = ID
            break
    # find the ending point
    end = sys.maxint
    for ID, frameDist in reversed(frames):
        if minDist < frameDist < maxDist:
            end = ID
            break
    frameList = [frame for frame in frames if frame[0] in range(start, end + 1)]
    return frameList[0::freq]


def main():
    # Process global variables and paths
    parse()
    global WORKDIR
    WORKDIR = os.getcwd()
    global MIN
    global MAX
    if args.min is None:
        MIN = 0
    else:
        MIN = args.min
    if args.max is None:
        MAX = sys.maxint
    else:
        MAX = args.max
    global FRAMESPATH
    if args.frames is not None and not os.path.isabs(args.frames):
        FRAMESPATH = WORKDIR + "/" + args.frames
    else:
        FRAMESPATH = args.frames
    if not os.path.isdir(FRAMESPATH):
        os.mkdir(FRAMESPATH)
    global DISTPATH
    if args.dist is not None and not os.path.isabs(args.dist):
        DISTPATH = WORKDIR + "/" + args.dist
    else:
        DISTPATH = args.dist

    # Read the distance file
    with open(DISTPATH, 'r') as distFile:
        frames = [(int(line.split()[0]), float(line.split()[1])) for line in distFile.readlines()]

    frameList = selectFrames(frames, MIN, MAX, args.freq)

    # Begin set-based comparison of collisions within frameset
    inSome = set()
    inAll = set()
    first = True
    totalFrames = len(frameList)
    # Find the collisions in each frame and add them to the sets
    for i, frame in enumerate(frameList):
        log("Frame %i of %i\n" % (i + 1, totalFrames))
        with open(FRAMESPATH + "/%i.pdb" % frame[0], 'r') as pdb:
            clashes = findClashes(pdb.readlines(), args.thres)
        inSome = inSome.union(set(clashes))
        if first:
            inAll = set(clashes)
            first = False
        else:
            inAll = inAll.intersection(set(clashes))
        log("Found %i collisions, %i total and %i conserved\n" %
            (len(clashes), len(inSome), len(inAll)))

    # Compute the transitory and conserved collisions across the frame set
    union = inSome.union(inAll)
    intersection = inSome.intersection(inAll)
    transitory = sorted(list(union - intersection), key=itemgetter(0))
    conserved = sorted(list(intersection), key=itemgetter(0))

    # Print output
    log("Found %i collisions\n" % (len(transitory) + len(conserved)))
    for item in conserved:
        sys.stdout.write("C %i %i\n" % (item[0], item[1]))
    sys.stdout.flush()
    for item in transitory:
        sys.stdout.write("T %i %i\n" % (item[0], item[1]))
    sys.stdout.flush()


def parse():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Analyze lineMD_RMSD output to compute "
                                        "changed clash residues")
    parser.add_argument("--min", help="start of distance range", type=float, default=0)
    parser.add_argument("--max", help="end of distance range", type=float, default=sys.maxint)
    parser.add_argument('-f', "--frames", help="folder containing PDBs of frames", type=str,
                        action=FullPath,
                        default="trajectory")
    parser.add_argument("--freq", help="only keep every n frames (default is 1 for all frames)",
                        type=int, default=1)
    parser.add_argument('-d', "--dist", help="two column frame/distance file", type=str,
                        action=FullPath,
                        default="distances")
    parser.add_argument('-t', "--thres", help="collision threshold in angstroms (default is 4)",
                        type=float, default=4.)
    global args
    args = parser.parse_args()

if __name__ == "__main__":
    main()
