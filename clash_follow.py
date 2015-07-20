#!/bin/env python
# clash_follow.py
# Analyze distances of discovered collisions over time

from argparse import ArgumentParser
from clash_screen import selectFrames
from stat import S_IEXEC
from shared import *
from numpy import linalg, array

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

    frameList = selectFrames(frames, MIN, MAX, args.freq)
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
                    if max(alldists) < args.maxthres:
                        return None
                    return clash, "TN", frameID, frameRMSD, atoms, alldists
        elif clashID < TtoNcount + CtoTcount:
            # C->T: print first negative collision
            for frameID, distance, atoms in frameResults:
                if distance > args.thres:  # no longer exists
                    frameRMSD = next(x[1] for x in frameList if x[0] == frameID)
                    printed += str(clash) + " CT %i %.3f " % (frameID, frameRMSD) + str(atoms) + "\n"
                    log(printed)
                    alldists = [item[1] for item in frameResults][0::args.outfreq]
                    if max(alldists) < args.maxthres:
                        return None
                    return clash, "CT", frameID, frameRMSD, atoms, alldists
        else:
            # C->N: print first negative collision
            for frameID, distance, atoms in frameResults:
                if distance > args.thres:  # no longer exists
                    frameRMSD = next(x[1] for x in frameList if x[0] == frameID)
                    printed += str(clash) + " CN %i %.3f " % (frameID, frameRMSD) + str(atoms) + "\n"
                    log(printed)
                    alldists = [item[1] for item in frameResults][0::args.outfreq]
                    if max(alldists) < args.maxthres:
                        return None
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
        # Write gnuplot output
        maxDist = int(5 * round(float(max([max(o[5]) for o in out])) / 5))  # maximum reached distance rounded to 5
        with open(args.plotfile, 'w') as plot:
            # write header
            plot.write("frame ")
            for (res1, res2), clashType, ID, RMSD, ato, dists in out:
                plot.write("%i/%i " % (res1, res2))
            plot.write("\n")
            for frame in xrange(len(out[0][5])):  # number of frames
                plot.write("%i " % frame)
                for res, clashType, ID, RMSD, ato, dists in out:
                    plot.write("%.3f " % dists[frame])
                plot.write("\n")
        CN = [o for o in out if o[1] == "CN"]
        CT = [o for o in out if o[1] == "CT"]
        TN = [o for o in out if o[1] == "TN"]
        with open(args.plotfile + "CN", 'w') as CNfile:
            # write header
            CNfile.write("frame ")
            for (res1, res2), clashType, ID, RMSD, ato, dists in CN:
                CNfile.write("%i/%i " % (res1, res2))
            CNfile.write("\n")
            for frame in xrange(len(CN[0][5])):  # number of frames
                CNfile.write("%i " % frame)
                for res, clashType, ID, RMSD, ato, dists in CN:
                    CNfile.write("%.3f " % dists[frame])
                CNfile.write("\n")
        with open(args.plotfile + "CT", 'w') as CTfile:
            # write header
            CTfile.write("frame ")
            for (res1, res2), clashType, ID, RMSD, ato, dists in CT:
                CTfile.write("%i/%i " % (res1, res2))
            CTfile.write("\n")
            for frame in xrange(len(CT[0][5])):  # number of frames
                CTfile.write("%i " % frame)
                for res, clashType, ID, RMSD, ato, dists in CT:
                    CTfile.write("%.3f " % dists[frame])
                CTfile.write("\n")
        with open(args.plotfile + "TN", 'w') as TNfile:
            # write header
            TNfile.write("frame ")
            for (res1, res2), clashType, ID, RMSD, ato, dists in TN:
                TNfile.write("%i/%i " % (res1, res2))
            TNfile.write("\n")
            for frame in xrange(len(TN[0][5])):  # number of frames
                TNfile.write("%i " % frame)
                for res, clashType, ID, RMSD, ato, dists in TN:
                    TNfile.write("%.3f " % dists[frame])
                TNfile.write("\n")

        # write gnuplot scripts
        def chunks(l, k):
            """Yield successive k-sized chunks from l."""
            for p in xrange(0, len(l), k):
                yield l[p:p + k]

        # Split the list into chunks
        lastID = 0
        for section, chunk in enumerate(list(chunks(out, args.max_plot))):
            # write a file for gnuplot commands
            with open("gnuplot_%i.sh" % section, 'w') as gnuplot:
                gnuplot.write("""echo "
set term png
set output 'gnuplot_%i.png'
set title 'All collisions part %i'
set key outside vertical right top maxcols 1;
set ylabel 'Distance (angstroms)'
set xlabel 'Frames'
set key autotitle columnhead
set yrange [0:%i]
plot '%s' using 1:%i w l, """ % (section, section + 1, maxDist, args.plotfile, lastID + 2))
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
            os.chmod("gnuplot_%i.sh" % section, os.stat("gnuplot_%i.sh" % section).st_mode | S_IEXEC)

        lastID = 0
        for section, chunk in enumerate(list(chunks(CN, args.max_plot))):
            # write a file for gnuplot commands
            with open("gnuplotCN_%i.sh" % section, 'w') as gnuplotCN:
                gnuplotCN.write("""echo "
set term png
set output 'gnuplotCN_%i.png'
set title 'Conserved to nonexistent collisions part %i'
set key outside vertical right top maxcols 1;
set ylabel 'Distance (angstroms)'
set xlabel 'Frames'
set key autotitle columnhead
set yrange [0:%i]
plot '%s' using 1:%i w l, """ % (section, section + 1, maxDist, args.plotfile + "CN", lastID + 2))
                # for each column
                for col in xrange(len(chunk) - 1):
                    gnuplotCN.write(" '' using 1:%i w l" % (col + 3 + lastID))
                    # comma if not last
                    if col < len(chunk) - 2:
                        gnuplotCN.write(", ")
                lastID += len(chunk)
                # end the script
                gnuplotCN.write("""
" | gnuplot -persist
""")
            # make executable
            os.chmod("gnuplotCN_%i.sh" % section, os.stat("gnuplotCN_%i.sh" % section).st_mode | S_IEXEC)

        lastID = 0
        for section, chunk in enumerate(list(chunks(CT, args.max_plot))):
            with open("gnuplotCT_%i.sh" % section, 'w') as gnuplotCT:
                gnuplotCT.write("""echo "
set term png
set output 'gnuplotCT_%i.png'
set title 'Conserved to transitory collisions part %i'
set key outside vertical right top maxcols 1;
set ylabel 'Distance (angstroms)'
set xlabel 'Frames'
set key autotitle columnhead
set yrange [0:%i]
plot '%s' using 1:%i w l, """ % (section, section + 1, maxDist, args.plotfile + "CT", lastID + 2))
                for col in xrange(len(chunk) - 1):
                    gnuplotCT.write(" '' using 1:%i w l" % (col + 3 + lastID))
                    if col < len(chunk) - 2:
                        gnuplotCT.write(", ")
                lastID += len(chunk)
                gnuplotCT.write("""
" | gnuplot -persist
""")
            os.chmod("gnuplotCT_%i.sh" % section, os.stat("gnuplotCT_%i.sh" % section).st_mode | S_IEXEC)

        lastID = 0
        for section, chunk in enumerate(list(chunks(TN, args.max_plot))):
            with open("gnuplotTN_%i.sh" % section, 'w') as gnuplotTN:
                gnuplotTN.write("""echo "
set term png
set output 'gnuplotTN_%i.png'
set title 'Transitory to nonexistent collisions part %i'
set key outside vertical right top maxcols 1;
set ylabel 'Distance (angstroms)'
set xlabel 'Frames'
set key autotitle columnhead
set yrange [0:%i]
plot '%s' using 1:%i w l, """ % (section, section + 1, maxDist, args.plotfile + "TN", lastID + 2))
                for col in xrange(len(chunk) - 1):
                    gnuplotTN.write(" '' using 1:%i w l" % (col + 3 + lastID))
                    if col < len(chunk) - 2:
                        gnuplotTN.write(", ")
                lastID += len(chunk)
                gnuplotTN.write("""
" | gnuplot -persist
""")
            os.chmod("gnuplotTN_%i.sh" % section, os.stat("gnuplotTN_%i.sh" % section).st_mode | S_IEXEC)


def parse():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Plot distances of discovered collisions over time")
    parser.add_argument('-f', "--frames", help="folder containing PDBs of frames", type=str,
                        action=FullPath, default="trajectory")
    parser.add_argument('-d', "--dist", help="two column frame/distance file", type=str,
                        action=FullPath, default="distances")
    parser.add_argument("--min", help="start of distance range", type=float, required=True)
    parser.add_argument("--max", help="end of distance range", type=float, required=True)
    parser.add_argument("--freq", help="only keep every n frames (default is 1 for all frames)",
                        type=int, default=1)
    parser.add_argument("--outfreq", help="same as freq, but for distance output", type=int, default=1)
    parser.add_argument("--max_plot", help="maximum number of collisions per plot", type=int, default=sys.maxint)
    parser.add_argument('-t', "--thres", help="collision threshold in angstroms (default is 4)",
                        type=float, default=4.)
    parser.add_argument("--maxthres", help="eliminate collisions that never go above this distance (default is 6)",
                        type=float, default=6.)
    parser.add_argument('-c', "--collisions", help="list of collisions "
                                                   "from clash_check", type=str, action=FullPath, default="check")
    parser.add_argument('--plotfile', help="output file prefix for gnuplot", type=str, action=FullPath, default="plot")
    global args
    args = parser.parse_args()


if __name__ == "__main__":
    main()
