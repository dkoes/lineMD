#!/bin/env python
# clash_follow.py
# Analyze distances of discovered collisions over time

from argparse import ArgumentParser
from numpy import linalg, array
from clash_screen import selectFrames
from shared import *

__author__ = 'Charles'


def main():
    # Process global variables and paths
    parse()
    global WORKDIR
    WORKDIR = os.getcwd()
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
    if args.collisions is not None and not os.path.isabs(args.collisions):
        COLLISIONSPATH = WORKDIR + "/" + args.collisions
    else:
        COLLISIONSPATH = args.collisions
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

    # Establish sets of collisions per type and read data
    TtoN, CtoT, CtoN = set(), set(), set()
    with open(COLLISIONSPATH, 'r') as collisionsFile:
        for line in collisionsFile.readlines():
            t = (int(line.split()[1]), int(line.split()[2]))
            if line.startswith("TN"):
                TtoN.add((min(t[0], t[1]), max(t[0], t[1])))
            elif line.startswith("CT"):
                CtoT.add((min(t[0], t[1]), max(t[0], t[1])))
            elif line.startswith("CN"):
                CtoN.add((min(t[0], t[1]), max(t[0], t[1])))
    TtoN, CtoT, CtoN = [sorted(list(s), key=lambda x: (x[0], x[1])) for s in (TtoN, CtoT, CtoN)]

    # Read the distance file
    with open(DISTPATH, 'r') as distFile:
        frames = [(int(line.split()[0]), float(line.split()[1])) for line in distFile.readlines()]

    if not args.check_all:
        frameList = selectFrames(frames, MIN, MAX, args.freq)
    else:
        frameList = frames[0::args.freq]
    totalFrames = len(frameList)

    # Set of all clashes, with three sublists sorted
    clashes = TtoN + CtoT + CtoN  # list of tuples
    TtoNcount, CtoTcount, CtoNcount = [len(s) for s in (TtoN, CtoT, CtoN)]
    totalClashes = TtoNcount + CtoTcount + CtoNcount
    log("Processing %i collisions\n" % totalClashes)

    # Find distances for each collision at different frames
    # Iterate over the frames to read residues
    log("Reading frame data.\n")
    frameResData = {}  # {frameID: residues, ...}
    resNames = {}

    for count, (ID, frameDist) in enumerate(frameList):
        log("\rFrame %i of %i" % (count + 1, totalFrames))
        with open(FRAMESPATH + "/%i.pdb" % ID, 'r') as pdb:
            # Read frame file into the residues dictionary
            residues = {}  # {residueID: [(atomID, atomCoords)]}
            for line in pdb.readlines():
                if line[0:4] != "ATOM" or line[17:20] not in RESIDUES:
                    continue
                vals = line.split()
                resID = int(vals[4])
                atomID = int(vals[1])
                if resID not in residues or residues[resID] is None:
                    residues[resID] = []
                atomCoords = [float(line[(30 + f * 8):(38 + f * 8)]) for f in range(3)]
                resName = vals[3]  # three-letter name of residue
                if resID not in resNames or resNames[resID] is None:
                    resNames[resID] = resName
                residues[resID].append((atomID, atomCoords))
            frameResData[count] = ID, residues
    log("\n")
    # Iterate over the collisions to check each frame
    log("Checking collisions.\n")

    def checkClash(clashID):
        """Returns a string representing the type and ID of the clash followed by the
        appropriate last frameID and RMSD and atoms/distance."""
        printed = "Transition %i of %i " % (clashID + 1, totalClashes)
        clash = clashes[clashID]
        # Iterate over each stored frame
        frameResults = []  # [(frameID, dist, atoms), ...]
        for frameCount, (frameID, frameResidues) in frameResData.iteritems():
            # find minimum distance between residues
            minDist = sys.maxint
            minAtoms = (0, 0)
            for a1, coord1 in frameResidues[clash[0]]:
                for a2, coord2 in frameResidues[clash[1]]:
                    thisDist = linalg.norm(abs(array(coord1) - array(coord2)))
                    if thisDist < minDist:
                        minDist = thisDist
                        minAtoms = a1, a2
            # Add the minimum distance for this clash
            frameResults.append((frameID, minDist, minAtoms))
        # Determine which frame to keep
        if clashID < TtoNcount:
            # T->N: print last positive collision
            for frameID, distance, atoms in reversed(frameResults):  # go backwards
                if distance < args.thres:  # clash exists
                    frameRMSD = next(x[1] for x in frameList if x[0] == frameID)
                    printed += str(clash) + " TN %i %.3f " % (frameID, frameRMSD) + str(atoms) + "\n"
                    log(printed)
                    alldists = [item[1] for item in frameResults][0::args.outfreq]
                    # eliminate it if it does not meet the max threshold at all
                    return clash, "TN", frameID, frameRMSD, atoms, alldists
        elif clashID < TtoNcount + CtoTcount:
            # C->T: print first negative collision
            for frameID, distance, atoms in frameResults:
                if distance > args.thres:  # no longer exists
                    frameRMSD = next(x[1] for x in frameList if x[0] == frameID)
                    printed += str(clash) + " CT %i %.3f " % (frameID, frameRMSD) + str(atoms) + "\n"
                    log(printed)
                    alldists = [item[1] for item in frameResults][0::args.outfreq]
                    return clash, "CT", frameID, frameRMSD, atoms, alldists
        else:
            # C->N: print first negative collision
            for frameID, distance, atoms in frameResults:
                if distance > args.thres:  # no longer exists
                    frameRMSD = next(x[1] for x in frameList if x[0] == frameID)
                    printed += str(clash) + " CN %i %.3f " % (frameID, frameRMSD) + str(atoms) + "\n"
                    log(printed)
                    alldists = [item[1] for item in frameResults][0::args.outfreq]
                    return clash, "CN", frameID, frameRMSD, atoms, alldists

    r = range(len(clashes))
    output = parMap(checkClash, r, n=(cpu_count() / 2), silent=True)
    if None in output:
        log(YELLOW + UNDERLINE + "Warning:" + END
            + " %i transitions not found in frames. --freq may have changed from clash_check"
              " or the transition may occur out of range.\n" % output.count(None))
    output = [o for o in output if o is not None]

    # write output
    # sort by type, then by RMSD, then by frameID, then by clash
    # output is clash, type, ID, RMSD, atoms, dists

    out = sorted([section for section in output if section is not None], key=lambda j: (j[1], j[3], j[2], j[0]))
    sys.stdout.write("# type resname1 res1 atom1 resname2 res2 atom2 frameID RMSD\n")
    sys.stdout.flush()
    for (res1, res2), clashType, ID, RMSD, (atom1, atom2), dists in out:
        sys.stdout.write("%s %s %i %i %s %i %i %i %.3f\n" % (clashType, resNames[res1], res1, atom1,
                                                             resNames[res2], res2, atom2, ID, RMSD))
    sys.stdout.flush()

    if args.plotfile is not None:

        def partition(c, i):
            """Separate the input list into two lists based on the condition"""
            tl = []
            fl = []
            for item in i:
                if c(item):
                    tl.append(item)
                else:
                    fl.append(item)
            return tl, fl

        # Separate based on type and whether it exceeds the minimum threshold at max distance
        (CNlow, CN), (CTlow, CT), (TNlow, TN) = [partition(lambda v: max(v[5]) < args.minthres,
                                                           [o for o in out if o[1] == y]) for y in ("CN", "CT", "TN")]

        plotArguments = ((out, "", "All"),
                         (CN, "CN", "Conserved to nonexistent"), (CNlow, "CNlow", "Conserved to nonexistent (low)"),
                         (CT, "CT", "Conserved to transitory"), (CTlow, "CTlow", "Conserved to transitory (low)"),
                         (TN, "TN", "Transitory to nonexistent"), (TNlow, "TNlow", "Transitory to nonexistent (low)"))
        plotArguments = [t for t in plotArguments if len(t[0]) > 0]  # exclude empty categories

        def chunks(q, k):
            """Yield successive k-sized chunks from q."""
            for p in xrange(0, len(q), k):
                yield q[p:p + k]

        maxDist = int(5 * round(float(max([max(o[5]) for o in out])) / 5))  # maximum reached distance rounded to 5

        for l, name, fullName in plotArguments:
            # Write gnuplot data
            with open(args.plotfile + name, 'w') as plot:
                # write header
                plot.write("RMSD ")
                for (res1, res2), clashType, ID, RMSD, ato, dists in l:
                    plot.write("%i/%i " % (res1, res2))
                plot.write("\n")
                for frame in xrange(len(l[0][5])):  # number of frames
                    plot.write("%.3f " % frameList[frame][1])
                    for res, clashType, ID, RMSD, ato, dists in l:
                        plot.write("%.3f " % dists[frame])
                    plot.write("\n")

            # write gnuplot scripts
            # Split the list into chunks
            lastID = 0
            chunksize = args.max_plot
            if name == "":
                chunksize = sys.maxint  # this is the "all" section, no need to chunk
            for section, chunk in enumerate(list(chunks(l, chunksize))):
                # write a file for gnuplot commands
                fileName = "gnuplot%s_%i.sh" % (name, section)
                with open(fileName, 'w') as gnuplot:
                    gnuplot.write("""echo "
set term png
set output 'gnuplot%s_%i.png'
""" % (name, section))
                    if name == "":
                        gnuplot.write("""set title '%s collisions'
set nokey
""" % fullName)
                    else:
                        gnuplot.write("""set title '%s collisions part %i'
set key autotitle columnhead outside vertical right top maxcols 1
""" % (fullName, section + 1))
                    gnuplot.write("""set ylabel 'Distance (angstroms)'
set xlabel 'RMSD (angstroms)'
set yrange [0:%i]
set xrange [0:*] reverse
""" % maxDist)
                    if MIN is not None and MIN > frameList[-1][1]:
                        gnuplot.write("""set arrow from %i,0 to %i,%i nohead lc rgb 'black'
""" % (MIN, MIN, maxDist))
                    if MAX is not None and MAX < frameList[0][1]:
                        gnuplot.write("""set arrow from %i,0 to %i,%i nohead lc rgb 'black'
""" % (MAX, MAX, maxDist))
                    gnuplot.write("""plot '%s' using 1:%i w l, """ % (args.plotfile + name, lastID + 2))
                    # for each column
                    for col in xrange(len(chunk) - 1):
                        gnuplot.write(" '' using 1:%i w l" % (col + 3 + lastID))
                        # comma if not last
                        if col < len(chunk) - 2:
                            gnuplot.write(", ")
                    lastID += len(chunk)

                    # end the script
                    gnuplot.write("""
" | gnuplot -persist
""")
        # make executable
        system("chmod +x gnuplot*.sh")


def parse():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Plot distances of discovered collisions over time")
    parser.add_argument('-f', "--frames", help="folder containing PDBs of frames", type=str,
                        action=FullPath, default="trajectory")
    parser.add_argument('-d', "--dist", help="two column frame/distance file", type=str,
                        action=FullPath, default="distances")
    parser.add_argument("--min", help="start of distance range", type=float, default=0)
    parser.add_argument("--max", help="end of distance range", type=float, default=sys.maxint)
    parser.add_argument("--check_all", help="follow the collision for the whole trajectory but mark at min and max", 
                        action="store_true")
    parser.add_argument("--freq", help="only keep every n frames (default is 1 for all frames)",
                        type=int, default=1)
    parser.add_argument("--outfreq", help="same as freq, but for distance output", type=int, default=1)
    parser.add_argument("--max_plot", help="maximum number of collisions per plot", type=int, default=sys.maxint)
    parser.add_argument('-t', "--thres", help="collision threshold in angstroms (default is 4)",
                        type=float, default=4.)
    parser.add_argument("--minthres", help="separate collisions that never go above this distance (default is 10)",
                        type=float, default=10.)
    parser.add_argument('-c', "--collisions", help="list of collisions "
                                                   "from clash_check", type=str, action=FullPath, default="check")
    parser.add_argument('--plotfile', help="output file prefix for gnuplot", type=str, action=FullPath, default="plot")
    global args
    args = parser.parse_args()


if __name__ == "__main__":
    main()
