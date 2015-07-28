#!/bin/env python
# traj_compute_dist.py
# Given a trajectory and its associated files, calculates the RMSD distance for each frame

from __future__ import division, print_function
from atom_tools import rmsdDist
from argparse import ArgumentParser
from shared import *
from lineMD_RMSD import log
from numpy import zeros
from shutil import copy
from process_traj import getTotalFrames

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

    global REFPATH
    if not REFPATH.endswith("pdb"):
        with open("ptraj.in", 'w') as script:
            script.write("parm %s\n" % REFPRMTOPPATH)
            script.write("trajin %s 1 1 1\n" % args.ref)
            script.write("trajout reference.pdb pdb\n")
        system("cpptraj < ptraj.in > /dev/null 2> /dev/null")
        REFPATH = WORKDIR + "/reference.pdb"

    calcRefCoords()

    global TOTALFRAMES
    TOTALFRAMES = getTotalFrames(PRMTOPPATH, COORDPATH)
    generatePDBs()


def parse():
    parser = ArgumentParser(description="Calculate the RMSD for each frame of a trajectory")
    parser.add_argument("--prmtop", '-p', help="AMBER topology file", type=str,
                        action=FullPath, required=True)
    parser.add_argument("--refprmtop", help="topology for reference file", type=str,
                        action=FullPath)
    parser.add_argument("--coord", '-c', help="NetCDF coordinates of entire trajectory",
                        type=str, action=FullPath, required=True)
    parser.add_argument("--ref", help="reference coordinate file in PDB format",
                        type=str, action=FullPath)
    parser.add_argument("--out", '-o', help="output folder for restart files", type=str,
                        action=FullPath, default="trajectory")
    parser.add_argument("--dist", '-d', help="output file for distance file",
                        type=str, action=FullPath, required=True)
    parser.add_argument("--precision", help="number of decimal places used in calculations. "
                                            "Note that certain calculations are "
                                            "limited by AMBER to 3 places.",
                        type=int, action="store", default=6)
    parser.add_argument("--log", help="log output file", type=str, action=FullPath)
    parser.add_argument("--segments", '-g',
                        help="Python list containing tuples representing segments to be "
                             "processed; each tuple specifies a begin and end residue for "
                             "the segment (inclusive)", type=str, action="store")
    parser.add_argument("--thres", '-t',
                        help="digits of accuracy for the convergence threshold (default is 3)",
                        type=float, default=3, action="store")
    global args
    args = parser.parse_args()


def prep():
    """Sets up variables for the entire script. Also detects whether analysis runs are necessary
    and returns a boolean to indicate this."""
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
    global COORDPATH
    global OUTPATH
    global DISTPATH
    global REFPRMTOPPATH
    global REFPATH

    # Path modifications
    if not os.path.isabs(args.prmtop):
        PRMTOPPATH = WORKDIR + "/" + args.prmtop
    else:
        PRMTOPPATH = args.prmtop
    if args.refprmtop is None:
        REFPRMTOPPATH = PRMTOPPATH
    elif args.refprmtop is not None and not os.path.isabs(args.refprmtop):
        REFPRMTOPPATH = WORKDIR + "/" + args.refprmtop
    else:
        REFPRMTOPPATH = args.refprmtop
    if not os.path.isabs(args.ref):
        REFPATH = WORKDIR + "/" + args.ref
    else:
        REFPATH = args.ref
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

    global SEGMENTS
    if args.segments is None:
        SEGMENTS = []
        log("Will process all segments.\n")
    else:
        error = RED + UNDERLINE + "Error:" + END + \
                RED + ' please provide a valid segment string.\n' + END
        SEGMENTS = eval(args.segments)
        if not isinstance(SEGMENTS, list):
            fail(error)
        for tup in SEGMENTS:
            if not isinstance(tup, tuple) or len(tup) != 2 \
                    or not isinstance(tup[0], int) \
                    or not isinstance(tup[1], int) or tup[0] > tup[1]:
                fail(error)
        log("Will process segments ")
        for tup in SEGMENTS:
            log("%i to %i; " % (tup[0], tup[1]))
        log("\n")


def calcRefCoords():
    """Reads the coordinates of the reference file into the global variable."""
    global REFCOORDS
    global SEGMENTS
    pdbLines = []
    with open(REFPATH) as pdb:
        if SEGMENTS is None:  # Process everything
            for pdbLine in pdb:
                # Skip non ATOM lines and non-alpha carbons
                if pdbLine[0:4] != "ATOM" or pdbLine[13:15] != "CA":
                    continue
                # Get line
                pdbLines.append(pdbLine)
        else:  # Process only segments
            for pdbLine in pdb:
                if pdbLine[0:4] != "ATOM" or pdbLine[13:15] != "CA":
                    continue
                resID = int(pdbLine[21:28])
                for tup in SEGMENTS:
                    if tup[0] <= resID <= tup[1]:
                        pdbLines.append(pdbLine)

    coords = zeros((len(pdbLines), 3), float)
    for index, line in enumerate(pdbLines):
        for j in range(3):
            coords[index, j] = float(line[(30 + j * 8):(38 + j * 8)])
    REFCOORDS = coords


def generatePDBs():
    """In a parallel method, generate all of the PDBs of the trajectory"""
    global TOTALFRAMES
    global OUTPATH

    with directory(OUTPATH):
        def generatePDB(frame):
            """Generate the PDB for a certain frame"""
            with open("frame_%i.in" % frame, 'w') as script:
                script.write("""parm %s
    trajin %s %i %i 1
    trajout frame_%i.pdb pdb
    """ % (PRMTOPPATH, COORDPATH, frame, frame, frame))
            system("cpptraj < frame_%i.in > /dev/null 2> /dev/null" % frame)
            with open("frame_%i.pdb" % frame) as pdb:
                thisDist = rmsdDist(pdbLines=list(pdb), refCoords=REFCOORDS, segments=SEGMENTS)
                log("Frame %i has dist %.3f\n" % (frame, thisDist))
                return frame, thisDist

        log("Generating PDB files.\n")
        distances = parMap(generatePDB, range(1, TOTALFRAMES + 1),
                           n=(cpu_count() / 2), silent=True)

        # clash_screen output sorted by ID
        distances = sorted(distances, key=itemgetter(0))

        # Write output file
        with open(DISTPATH, 'w') as dist_vs_t:
            for frameID, dist in distances:
                dist_vs_t.write("%i %.3f\n" % (frameID, dist))

        # Name PDB files
        copy("frame_1.pdb", "initial.pdb")

        for frameID, dist in distances:
            os.rename("frame_%i.pdb" % frameID, "%i.pdb" % frameID)
            unblocked_system("rm frame_%i.in" % frameID)
    unblocked_system("rm reference.pdb")

if __name__ == "__main__":
    main()
