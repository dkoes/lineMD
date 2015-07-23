#!/bin/env python
# clash_frame_gen.py
# Processes lineMD_RMSD output for clash screening

from __future__ import division, print_function
from argparse import ArgumentParser
from glob import glob
import shutil
from shared import *


__author__ = 'Charles Yuan'


def generatePDB(run, frameID):
    """Generate the PDB for the last frame in a certain run and move it
    to location frameID in the output"""
    with directory(run):
        decompress("end.rst.gz")
        with open("clash.in", 'w') as script:
            script.write("""parm %s
    trajin end.rst 1 1 1
    trajout end.pdb pdb
    """ % PRMTOPPATH)
        system("cpptraj < clash.in > /dev/null 2> /dev/null")
        compress("end.rst")
        unblocked_system("rm clash.in")
        shutil.move("end.pdb", FRAMESPATH + "/%i.pdb" % frameID)


def main():
    # Process global variables and paths
    parse()
    global WORKDIR
    WORKDIR = os.getcwd()
    global PRMTOPPATH
    if args.prmtop is not None and not os.path.isabs(args.prmtop):
        PRMTOPPATH = WORKDIR + "/" + args.prmtop
    else:
        PRMTOPPATH = args.prmtop
    if not os.path.isfile(PRMTOPPATH):
        fail(RED + UNDERLINE + "Error:" + END + RED +
             " specified topology file does not exist.\n" + END)
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
    global LINEPATH
    if args.line is not None and not os.path.isabs(args.line):
        LINEPATH = WORKDIR + "/" + args.line
    else:
        LINEPATH = args.line

    # read all runs in the directory
    runs = []  # [(runPath, dist), ...]
    for runPath in glob(LINEPATH + "/C*_*/R*"):
        with open(runPath + "/run_info", 'r') as runInfo:
            runs.append((runPath, float(runInfo.readlines()[1].split()[1])))
    runs.sort(key=itemgetter(1))
    runs.reverse()  # RMSD is high-to-low

    # select and splice runs according to parameters
    if args.min is not None and args.max is not None:
        runs = [run for run in runs if args.min < run[1] < args.max]
    runs = runs[0::args.freq]

    totalRuns = len(runs)
    log("Processing %i runs.\n" % totalRuns)

    if args.override:
        open(DISTPATH, 'w').close()

    # Generate each PDB and write to file
    with open(DISTPATH, 'a') as dist:
        def processRun(ID):
            """Wrapper for generatePDB"""
            path, runDist = runs[ID]
            log(str(ID) + " " + path + "\n")
            if not args.dist_only:
                generatePDB(path, ID)
            # Temporarily write ID, dist to the dist file, to be rewritten later
            dist.write(str(ID) + " " + str(runDist) + "\n")
            dist.flush()
        # Generate the PDB and/or write to file in parallel
        parMap(processRun, range(0, totalRuns), n=(cpu_count() / 2))

    # Calculate distances and write dist file
    dists = []  # [(frameID, dist)...]
    # Read existing lines in the dist file and sort them
    with open(DISTPATH, 'r') as distFile:
        for line in distFile.readlines():
            # read ID, dist
            dists.append((int(line.split()[0]), float(line.split()[1])))
        dists.sort(key=itemgetter(1))
        dists.reverse()

    # Write new lines
    with open(DISTPATH, 'w') as distFile:
        for frameID, dist in dists:
            distFile.write(str(frameID) + " " + str(dist) + "\n")
        distFile.flush()
    log("\rProcessing complete.\n")


def parse():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Process lineMD_RMSD output for clash screening")
    parser.add_argument('-p', "--prmtop", help="topology file", type=str, action=FullPath)
    parser.add_argument('-l', "--line", help="folder containing lineMD output", type=str,
                        action=FullPath, default="line")
    parser.add_argument('-f', "--frames", help="folder for frame output", type=str,
                        action=FullPath, default="trajectory")
    parser.add_argument('-d', "--dist", help="output file path", type=str, action=FullPath,
                        default="distances")
    parser.add_argument("--dist_only", help="only export distance information, not frames",
                        action="store_true")
    parser.add_argument("--min", help="start of distance range", type=float, default=0)
    parser.add_argument("--max", help="end of distance range", type=float, default=sys.maxint)
    parser.add_argument("--freq", help="only keep every n frames (default is 1 for all frames)",
                        type=int, default=1)
    parser.add_argument('-O', "--override", help="override output", action="store_true")
    global args
    args = parser.parse_args()

if __name__ == "__main__":
    main()
