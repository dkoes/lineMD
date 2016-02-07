#!/bin/env python
# process_traj.py
# For a completed lineMD trajectory, generate distance files for PyBrella

from __future__ import division, print_function
from argparse import ArgumentParser
from atom_tools import calcCenterAtoms, atomDist
from shared import *
from shutil import copy

__author__ = "Charles Yuan"
__credits__ = ["Charles Yuan", "David Koes", "Matthew Baumgartner"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Charles Yuan"
__email__ = "charlesyuan314@gmail.com"
__status__ = "Development"


def main():
    """Calls major subroutines, prepares files, and prints messages.
    Hands control over to eventLoop afterwards."""
    parse()
    prep()
    log("Running on %i cores.\n" % cpu_count())
    global TOTALFRAMES
    if not args.skip:
        # Process the PDB files for the whole trajectory
        TOTALFRAMES = getTotalFrames(PRMTOPPATH, COORDPATH)
        generatePDBs()
    else:
        TOTALFRAMES = max([int(name[6:][:-4]) for name in os.listdir(OUTPATH)
                           if name.startswith("frame")])
        log("Trajectory has %i frames.\n" % TOTALFRAMES)

    # Calculate the distances for all the frames
    distances = calcDists()

    # Requires PyBrella output, sort by distance and make unique output
    distances = sorted(distances, key=itemgetter(1))
    newDistances = []
    # Processing in order to only include first frame of a given distance
    prevDist = sys.maxint
    for frameID, dist in distances:
        formattedDist = "%.3f" % dist
        if frameID == 0:
            prevDist = formattedDist
            newDistances.append((frameID, dist))
            continue
        if formattedDist != prevDist:
            prevDist = formattedDist
            newDistances.append((frameID, dist))

    # Write output file
    with open(DISTPATH, 'w') as dist_vs_t:
        for frameID, dist in newDistances:
            dist_vs_t.write('{:>8}'.format("%i" % frameID))
            dist_vs_t.write("    %.3f\n" % dist)
    if not args.pdb:
        unblocked_system("rm %s/frame_*.pdb" % OUTPATH)
    if args.rst:
        generateRSTs(sorted([item[0] for item in newDistances]))
    # PyBrella output named by distance
    if args.rst:
        for frameID, dist in newDistances:
            os.rename("%s/frame_%i.rst" % (OUTPATH, frameID), "%s/%.3f.rst" % (OUTPATH, dist))
    if args.pdb:
        for frameID, dist in newDistances:
            os.rename("%s/frame_%i.pdb" % (OUTPATH, frameID), "%s/%.3f.pdb" % (OUTPATH, dist))


def parse():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Generate distance files for PyBrella")
    parser.add_argument("--prmtop", '-p', help="AMBER topology file", type=str, action=FullPath,
                        required=True)
    parser.add_argument("--coord", '-c', help="NetCDF coordinates of entire trajectory", type=str,
                        action=FullPath, required=True)
    parser.add_argument("--pdb", help="export PDBs", action="store_true")
    parser.add_argument("--rst", help="export RSTs", action="store_true")
    parser.add_argument("--out", '-o', help="output folder for frames", type=str,
                        action=FullPath, default="trajectory")
    parser.add_argument("--dist", '-d', help="output file for distance file", type=str,
                        action=FullPath, default="sorted_dist_vs_t")
    parser.add_argument("--precision", help="number of decimal places used in calculations. "
                                            "Note that certain calculations are limited by "
                                            "AMBER to 3 places.",
                        type=int, action="store", default=6)
    parser.add_argument("--skip", action="store_true")
    global args
    args = parser.parse_args()


def prep():
    """Sets up variables for the entire script. Also detects whether analysis runs are necessary and
    returns a boolean to indicate this."""
    global WORKDIR
    WORKDIR = os.getcwd()
    # Basic verifications
    if args.prmtop is None or args.coord is None:
        fail(RED + UNDERLINE + "Error:" + END + RED +
             " please provide the topology and coordinate files.\n" + END)
    if not os.path.splitext(args.coord)[1].lower() == ".nc":
        fail(RED + UNDERLINE + "Error:" + END + RED +
             " coordinate file extension is invalid. Please specify a binary NetCDF file.\n" + END)

    global PRMTOPPATH
    global OUTPATH
    global DISTPATH

    # Path modifications
    if not os.path.isabs(args.prmtop):
        PRMTOPPATH = WORKDIR + "/" + args.prmtop
    else:
        PRMTOPPATH = args.prmtop
    if not os.path.isabs(args.coord):
        COORDPATH = WORKDIR + "/" + args.coord
    else:
        COORDPATH = args.coord
    if not os.path.isabs(args.out):
        OUTPATH = WORKDIR + "/" + args.out
    else:
        OUTPATH = args.out
    if not os.path.isdir(OUTPATH):
        os.mkdir(OUTPATH)
    if not os.path.isabs(args.dist):
        DISTPATH = WORKDIR + "/" + args.dist
    else:
        DISTPATH = args.dist

    global TOTALFRAMES
    TOTALFRAMES = 0


def getTotalFrames(prmtopPath, coordPath):
    """Calculates the total number of frames and sets the TOTALFRAMES variable."""
    with open("test.in", 'w') as script:
        script.write("""parm %s
trajin %s 1 1 1
trajout test.nc netcdf
""" % (prmtopPath, coordPath))
    system("cpptraj < test.in > test.out")
    frameLine = [line for line in open("test.out") if "reading" in line][0]
    if frameLine is None:
        fail(RED + "Error: could not get number of frames.\n" + END)
    totalFrames = int(frameLine.split()[15][:-1])
    log("Trajectory has %i frames.\n" % totalFrames)
    unblocked_system("rm test.in")
    unblocked_system("rm test.out")
    unblocked_system("rm test.nc")
    return totalFrames


def generatePDBs():
    """In a parallel method, generate all of the PDBs of the trajectory"""
    global TOTALFRAMES
    global OUTPATH
    os.chdir(OUTPATH)

    def generatePDB(frame):
        """Generate the PDB for a certain frame"""
        with open("frame_%i.in" % frame, 'w') as script:
            script.write("""parm %s
trajin %s %i %i 1
trajout frame_%i.pdb pdb
""" % (PRMTOPPATH, COORDPATH, frame, frame, frame))
        system("cpptraj < frame_%i.in > /dev/null 2> /dev/null" % frame)
        unblocked_system("rm frame_%i.in" % frame)
        # noinspection PyUnresolvedReferences
        log(("\rCompleted frame %i." % frame).ljust(getTerminalWidth()))

    log("Generating PDB files.\n")
    parMap(generatePDB, range(1, TOTALFRAMES + 1), n=(cpu_count() / 2))
    log("\n")
    copy("frame_1.pdb", "initial.pdb")
    os.chdir(WORKDIR)


def generateRSTs(frames):
    """In a parallel method, generate the RSTs of the trajectory specified by frames"""
    global TOTALFRAMES
    global OUTPATH
    os.chdir(OUTPATH)

    def generateRST(frame):
        """Generate the RST for a certain frame"""
        with open("frame_%i.in" % frame, 'w') as script:
            script.write("""parm %s
trajin %s %i %i 1
trajout frame_%i.rst restart
""" % (PRMTOPPATH, COORDPATH, frame, frame, frame))
        system("cpptraj < frame_%i.in > /dev/null 2> /dev/null" % frame)
        unblocked_system("rm frame_%i.in" % frame)
        # noinspection PyUnresolvedReferences
        log(("\rCompleted frame %i." % frame).ljust(getTerminalWidth()))

    log("Generating RST files.\n")
    parMap(generateRST, frames, n=(cpu_count() / 2))
    log("\n")
    copy("frame_1.rst", "initial.rst")
    os.chdir(WORKDIR)


def calcDists():
    """In a parallel method, calculate all of the distances for each frame. 
    Returns a list of (frame, distance)"""
    global TOTALFRAMES
    os.chdir(OUTPATH)

    with open(OUTPATH + "/initial.pdb") as p:
        LIGANDATOM, PROTEINATOM, lAtomCoord, pAtomCoord, lCenter, pCenter = \
            calcCenterAtoms(list(p))
    log("Selected ligand center atom ID " + MAGENTA + "%i" % LIGANDATOM + END +
        " at (%.3f, %.3f, %.3f), " % (lAtomCoord[0], lAtomCoord[1], lAtomCoord[2]) +
        "closest to (%.3f, %.3f, %.3f).\n" % (lCenter[0], lCenter[1], lCenter[2]))
    log("Selected protein center atom ID " + MAGENTA + "%i" % PROTEINATOM + END +
        " at (%.3f, %.3f, %.3f), " % (pAtomCoord[0], pAtomCoord[1], pAtomCoord[2]) +
        "closest to (%.3f, %.3f, %.3f).\n" % (pCenter[0], pCenter[1], pCenter[2]))

    def calcDist(frame):
        """Calculate the distance for a certain frame. Returns (frame, distance)."""
        with open("frame_%i.pdb" % frame) as pdb:
            dist = atomDist(list(pdb), LIGANDATOM, PROTEINATOM)
        log("Frame %i has distance %.3f\n" % (frame, dist))
        return frame, dist

    log("Calculating distances.\n")
    output = parMap(calcDist, range(1, TOTALFRAMES + 1), n=cpu_count() / 2)
    os.chdir(WORKDIR)
    return output

if __name__ == "__main__":
    main()
