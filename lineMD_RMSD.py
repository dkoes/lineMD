#!/bin/env python
# lineMD_RMSD.py
# lineMD by Charles Yuan, 2015 (RMSD version)
# Executes linear MD simulations with AMBER in order to generate ligand motion
# trajectories without the bias of steered molecular dynamics.
# Based on SpaceMD by Matthew Baumgartner and Nick Pabon
# Contains modified code from the pdbTools project by Mike Harms: https://code.google.com/p/pdb-tools/
# and from rmsd by Jimmy Charnley Kromann: https://github.com/charnley/rmsd/

from __future__ import division, print_function
from argparse import ArgumentParser, SUPPRESS
from atom_tools import rmsdDist
import copy
from glob import glob
from lineMD import eventLoop, determineSplit, getFinishedRuns, stitchTrajectory  # and exportRestarts
from numpy import zeros
from operator import attrgetter
import random
from shared import *
import shutil
from stat import S_IEXEC
from time import strftime

__author__ = "Charles Yuan"
__license__ = "GPL"
__version__ = "2.0"
__email__ = "charlesyuan314@gmail.com"
__status__ = "Development"


def main():
    """Calls major subroutines, prepares files, and prints messages. Hands control over to eventLoop afterwards."""
    parse()

    log(CYAN + BOLD + "lineMD" + END + CYAN + " version " + MAGENTA + str(__version__) + CYAN + " starting up on "
        + MAGENTA + "%s.\n" % args.queue_name + END)
    log("Configured for " + MAGENTA + str(args.precision) + END + " decimal places.\n" + END)
    log("Using topology at " + UNDERLINE + "%s\n" % args.prmtop + END)
    log("Using coordinates at " + UNDERLINE + "%s\n" % args.coord + END)
    log("Using reference topology at " + UNDERLINE + "%s\n" % args.refprmtop + END)
    log("Using reference coordinates at " + UNDERLINE + "%s\n" % args.ref + END)
    log("Using " + MAGENTA + str(args.steps) + END + " timesteps per simulation.\n")
    log(MAGENTA + str(args.sample) + END + " samples will be taken per run.\n")
    if args.restart_out is not None:
        log(MAGENTA + str(args.frame) + END + " frames will be in every run upon completion.\n")
    log(BLUE + "Simulation will end when RMSD is within " + MAGENTA + "%.*f" % (args.precision, args.min) + BLUE +
        " angstroms.\n" + END)
    log(BLUE + "Runs past " + MAGENTA + "%.*f" % (args.precision, args.max) + BLUE +
        " angstroms in the opposite direction will be rejected.\n" + END)
    log("Dynamic explored counts are ")
    if args.adjust:
        log(MAGENTA + "enabled.\n" + END)
    else:
        log(MAGENTA + "disabled.\n" + END)

    runInit = not prep()

    global COORDPATH

    if not os.path.isfile(WORKDIR + "/init.rst.gz"):
        system("gzip -c %s > init.rst.gz" % COORDPATH)
    COORDPATH = WORKDIR + "/init.rst.gz"

    if not os.path.isfile(WORKDIR + "/reference.pdb"):
        with open("ptraj.in", 'w') as script:
            script.write("parm %s\n" % REFPRMTOPPATH)
            script.write("trajin %s 1 1 1\n" % args.ref)
            script.write("trajout reference.pdb pdb\n")
        system("cpptraj < ptraj.in > /dev/null 2> /dev/null")

    calcRefCoords()

    if args.stitch:
        log("Reading cluster information.\n")
        readClusterInfo(readExplored=False)  # print, explored count is unnecessary
        stitchTrajectory()
        sys.exit(0)

    if runInit:
        init()
    if args.migrate:
        log("Reading cluster information.\n")
        readClusterInfo(silent=True, readInfo=False, readRuns=False, readExplored=False)  # Read nothing
        determineSplit()
        readClusterInfo(readExplored=False)  # do not read explored
        # move runs for each cluster
        for cluster in sorted([c for c in CLUSTERS.values() if c.ID != 'R'], key=attrgetter("dist")):
            migrateRuns([run.ID for run in cluster.runs.values()], cluster)
    if not os.path.isdir(WORKDIR + "/CR"):
        fail(RED + UNDERLINE + "Error:" + END + RED + " the running cluster is missing.\n" + END)
    elif not runInit:
        finishedRuns = getFinishedRuns()
        if finishedRuns:
            analysis(finishedRuns)
        else:
            analysis([])
    if args.loop > 0:
        eventLoop()


def parse():
    """Prepare the argument parser."""
    parser = ArgumentParser(description="execute linear MD simulations with AMBER.")
    parser.add_argument("--prmtop", '-p', help="AMBER topology file", type=str, action=FullPath, required=True)
    parser.add_argument("--coord", '-c', help="restart file from equilibration", type=str, action=FullPath)
    parser.add_argument("--ref", help="reference coordinate file at endpoint", type=str, action=FullPath, required=True)
    parser.add_argument("--refprmtop", help="topology for reference file", type=str, action=FullPath)
    parser.add_argument("--min", help="endpoint RMSD", action="store", type=float)
    parser.add_argument("--max", help="maximum RMSD change permitted (traveling in incorrect direction)",
                        action="store", type=float)
    parser.add_argument("--bin", '-b',
                        help="width, in angstroms, per bin. "
                             "Ensure that both max and min are multiples of this width, and do not change"
                             " even if splits have occurred.", action="store", type=float)
    parser.add_argument("--steps", '-s', help="nstlim value for AMBER config (default is 50000). Should be a "
                                              "multiple of the \"--sample\" and \"--frame\" parameters.",
                        type=int, action="store", default=50000)
    parser.add_argument("--sample", '-w', help="number of sample frames desired per run during execution "
                                               "(default is 100)", type=int, action="store", default=100)
    parser.add_argument("--frame", '-f', help="number of frames desired per run in the output (default is 100)",
                        type=int, action="store", default=100)
    parser.add_argument("--threads", '-t', help="number of simultaneous runs", type=int, action="store", default=1)
    parser.add_argument("--queue_name", '-q', help="queue name", type=str, required=True)
    parser.add_argument("--loop", '-l', help="number of seconds before the loop checks status. "
                                             "Set to 0 to disable the loop.", type=int, action="store", default=30)
    parser.add_argument("--migrate", help=SUPPRESS, action="store_true")
    parser.add_argument("--split", help=SUPPRESS, action="store_true")
    parser.add_argument("--adjust", help="use the dynamic explored count (experimental)", action="store_true")
    parser.add_argument("--precision", help="number of decimal places used in calculations. "
                                            "Increase to allow more splits. Less than 8 recommended.",
                        type=int, action="store", default=6)
    parser.add_argument("--stitch", help="do nothing but stitch the trajectory", action="store_true")
    parser.add_argument("--trash", help="specify this directory to hold runs that have been deleted", type=str,
                        action=FullPath)
    parser.add_argument("--log", '-o', help="log output file", type=str, action=FullPath)
    parser.add_argument("--restart_out", '-r', help="specify this path to save the restarts from the final trajectory"
                                                    " and stitch longer trajectories using the --frame parameter",
                        type=str, action=FullPath)
    parser.add_argument("--segments", '-g', help="Python list containing tuples representing segments to be processed;"
                                                 " each tuple specifies a begin and end residue for the segment "
                                                 "(inclusive)",
                        type=str, action="store")
    global args
    args = parser.parse_args()


def prep():
    """Sets up variables for the entire script. Also detects whether analysis runs are necessary and
    returns a boolean to indicate this."""
    global WORKDIR
    WORKDIR = os.getcwd()

    global PAUSE
    if args.loop <= 0:
        PAUSE = 1
    else:
        PAUSE = args.loop

    global BINWIDTH
    BINWIDTH = round(args.bin, args.precision)

    global RUNNING  # Number of threads currently running
    RUNNING = 0

    global THREADS
    THREADS = args.threads

    global CLUSTERS  # Dictionary of Cluster IDs and clusters
    CLUSTERS = {}

    global RUNANALYSIS
    RUNANALYSIS = False
    if os.path.exists(WORKDIR + "/C0_0") or os.path.exists(WORKDIR + "/CR"):
        RUNANALYSIS = True

    # Basic verifications
    if args.prmtop is None or args.coord is None:
        fail(RED + UNDERLINE + "Error:" + END + RED + " please provide the topology and coordinate files.\n" + END)
    if not os.path.splitext(args.coord)[1].lower() == ".rst":
        fail(RED + UNDERLINE + "Error:" + END + RED +
             " coordinate file extension is invalid. Please specify a formatted RST file.\n" + END)
    if RUNANALYSIS and args.max is None:
        fail(RED + UNDERLINE + "Error:" + END + RED + " Please provide the maximum RMSD.\n" + END)
    if RUNANALYSIS and args.min is None:
        fail(RED + UNDERLINE + "Error:" + END + RED + " Please provide the minimum RMSD.\n" + END)

    global PRMTOPPATH
    global REFPRMTOPPATH
    global COORDPATH
    global TRASHPATH
    global RESTARTPATH

    # Path modifications
    if args.prmtop is not None and not os.path.isabs(args.prmtop):
        PRMTOPPATH = WORKDIR + "/" + args.prmtop
    else:
        PRMTOPPATH = args.prmtop
    if args.refprmtop is None:
        REFPRMTOPPATH = PRMTOPPATH
    elif args.refprmtop is not None and not os.path.isabs(args.refprmtop):
        REFPRMTOPPATH = WORKDIR + "/" + args.refprmtop
    else:
        REFPRMTOPPATH = args.refprmtop
    if args.coord is not None and not os.path.isabs(args.coord):
        COORDPATH = WORKDIR + "/" + args.coord
    else:
        COORDPATH = args.coord
    if args.restart_out is not None and not os.path.isabs(args.restart_out):
        RESTARTPATH = WORKDIR + "/" + args.restart_out
    else:
        RESTARTPATH = args.restart_out
    if args.restart_out is not None and not os.path.isdir(RESTARTPATH):
        os.mkdir(RESTARTPATH)
    if args.trash is not None and not os.path.isabs(args.trash):
        TRASHPATH = WORKDIR + "/" + args.trash
    else:
        TRASHPATH = args.trash
    if args.trash is not None and not os.path.isdir(TRASHPATH):
        os.mkdir(TRASHPATH)

    global NOPROGRESS  # Number of times returned to the initial bin
    NOPROGRESS = 0

    global NOPROGRESSCUTOFF  # If we have to return to the initial bin this many times, split the bins
    NOPROGRESSCUTOFF = 10

    if NOPROGRESSCUTOFF < THREADS:
        NOPROGRESSCUTOFF = THREADS

    global SPLIT  # Number of times we have split the bins
    SPLIT = 0

    global SPLITMAX  # Maximum number of times permitted for splitting bins.
    # Determine SPLITMAX; this code is still experimental
    tempBinWidth = BINWIDTH
    SPLITMAX = 0
    while True:
        oldBinWidth = tempBinWidth
        tempBinWidth = round(tempBinWidth / 2.0, args.precision)
        if oldBinWidth == 2 * tempBinWidth:  # Divide and round, then multiply. If not equal, then splitting should end
            SPLITMAX += 1
        else:
            break

    SPLITMAX -= 1

    if SPLITMAX > 4:
        SPLITMAX = 4

    if SPLITMAX > 0:
        log("Splitting will occur for " + MAGENTA + str(SPLITMAX) + END + " times maximum.\n")
    else:
        log("Splitting will not occur on such a small bin width.\n")

    if args.steps / args.sample != int(args.steps / args.sample) or \
       args.steps / args.frame != int(args.steps / args.frame):
        log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW +
            " the sample or frame frequencies (--sample or --frame) do not divide evenly into the "
            "total number of simulation timesteps (--steps). Proceed with caution.\n" + END)

    if args.max / args.bin != int(args.max / args.bin) or \
       args.min / args.bin != int(args.min / args.bin):
        log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW +
            " the bin size (--bin) does not divide evenly into the "
            "max or min distance (--max or --min). Proceed with caution.\n" + END)

    global SEGMENTS
    if args.segments is None:
        SEGMENTS = []
        log("Will process all segments.\n")
    else:
        error = RED + UNDERLINE + "Error:" + END + RED + " please provide a valid segment string.\n" + END
        SEGMENTS = eval(args.segments)
        if not isinstance(SEGMENTS, list):
            fail(error)
        for tup in SEGMENTS:
            if not isinstance(tup, tuple) or len(tup) != 2 \
                    or not isinstance(tup[0], int) or not isinstance(tup[1], int) or tup[0] > tup[1]:
                fail(error)
        log("Will process segments ")
        for tup in SEGMENTS:
            log("%i to %i; " % (tup[0], tup[1]))
        log("\n")
    return RUNANALYSIS


class Run(object):
    """A Run object holds its own topology, input, output, and coordinate files. It can calculate its own
     distance and knows its Cluster. Its folder format is R? where ? is the ID of the run. It can execute itself
     with the execute() method. The folder holds a "run_info" file to store the path of the run from which input
     coordinates were copied. A folder and run_info will automatically be created."""

    _ID = 0  # chronologically assigned ID number, also part of name of folder
    _UID = 0  # permanent ID, assigned on a rolling basis.
    _frame = 0  # the number of frames in the coordinate file after processing
    # Note: always provide the UID as Run.getNextUID() when initializing the Run,
    # unless the Run can immediately readInfo() or create() with a run_info file.
    _clusterID = '0_0'  # the cluster with which this run is associated. May be a number or 'R'
    _previous = 0  # the UID of the previous run, will be "initial" for the initial run
    _explored = 0  # the number of times this run has been copied
    _dist = 0  # distance of the ligand from the protein

    def __init__(self, ID=0, UID=0, clusterID='0_0', previous=None, explored=0, frame=0):
        self._ID = int(ID)
        self._clusterID = str(clusterID)
        self._previous = previous
        self._UID = int(UID)
        self._explored = int(explored)
        self._frame = int(frame)

    def __str__(self):
        return "<Run: ID %i, UID %i, cluster %s>" % (self._ID, self._UID, self._clusterID)

    @property
    def path(self):  # directory holding trajectory files
        return "%s/C%s/R%i" % (WORKDIR, self._clusterID, self._ID)

    @property
    def shortPath(self):
        return "C%s/R%i" % (self._clusterID, self._ID)

    @property
    def ID(self):
        return self._ID

    @property
    def UID(self):
        return self._UID

    @property
    def clusterID(self):
        return self._clusterID

    @property
    def previous(self):
        return self._previous

    @property
    def explored(self):
        return self._explored

    @explored.setter
    def explored(self, explored):
        self._explored = explored

    @staticmethod
    def getNextUID():
        """Read the next UID from the currentUID file and update the file."""
        newUID = 0  # default
        # Try opening the existing currentUID file
        try:
            with open(WORKDIR + "/currentUID") as currentUID:
                info = []
                for line in currentUID:
                    info.append(int(line))
                newUID = info[0] + 1
        except IOError:
            pass  # No existing file
        # Write a new currentUID file
        with open(WORKDIR + "/currentUID", 'w') as currentUID:
            currentUID.write(str(newUID) + '\n')
        # Set the UID
        return newUID

    @staticmethod
    def move(r, c):
        """Move the run r from wherever it is to the cluster c, compressing files if necessary. Returns the new run."""
        if r.clusterID == c.ID:  # Already in that cluster
            return r
        # Find the next appropriate ID
        runDirectories = [int(name[1:]) for name in os.listdir(c.path) if
                          os.path.isdir(os.path.join(c.path, name))]
        if runDirectories:
            newID = max(runDirectories) + 1
        else:
            newID = 0
        new = Run(ID=newID, clusterID=c.ID)  # No UID because we will create with run_info
        coord = None
        outFile = None
        inFile = None
        info = None
        endRestart = None
        beginRestart = None
        with directory(r.path):
            if os.path.isfile("end.rst"):
                compress("end.rst")
            if os.path.isfile("end.rst.gz"):
                endRestart = r.path + "/end.rst.gz"
            if os.path.isfile("coord.nc"):
                compress("coord.nc")
            if os.path.isfile("coord.nc.gz"):
                coord = r.path + "/coord.nc.gz"
            if os.path.isfile("line.out"):
                compress("line.out")
            if os.path.isfile("line.out.gz"):
                outFile = r.path + "/line.out.gz"
            if os.path.isfile("line.in"):
                inFile = r.path + "/line.in"
            if os.path.isfile("begin.rst"):
                compress("begin.rst")
            if os.path.isfile("begin.rst.gz"):
                beginRestart = r.path + "/begin.rst.gz"
            if os.path.isfile("run_info"):
                info = r.path + "/run_info"
            else:
                fail(RED + UNDERLINE + "Error:" + END + RED + " run at %s has no run_info.\n" % r.path + END)
        new.create(endRestart=endRestart, info=info, coord=coord, outFile=outFile, inFile=inFile,
                   beginRestart=beginRestart)
        # Delete the old run
        r.delete(trash=False)
        # Change Cluster object data
        if new.ID in c.runs:
            fail(RED + UNDERLINE + "Error:" + END + RED + " the new run is already in the dictionary.\n" + END)
        c.addRun(ID=new.ID, run=new)
        return new

    def execute(self):
        """Call a run from its script"""
        global RUNNING
        if RUNNING >= THREADS:  # Reached maximum, abandon this run
            self.delete(trash=False)
            return

        with directory(self.path):
            if not os.path.isfile("./run.sh"):
                log(RED + UNDERLINE + "Error:" + END + RED + " run.sh does not exist at %s.\n" % self.path + END)

            system("./run.sh >> out 2>&1")

            RUNNING += 1  # Increment the global counter
        return

    def check(self, coordName):
        """Verify that the run's ending coordinates are present"""
        with directory(self.path):
            if os.path.isfile(coordName):
                size = os.stat(coordName)[6]
                if size > 22 + len(coordName):  # minimal file size for compressed files
                    return True
                else:
                    return False  # file is empty
            else:
                return False  # file does not exist

    def processDist(self):
        """Calculates the distance between the protein and ligand at the final frame and processes coordinate files.
        Returns (dist, frame) of the smallest dist
        """

        if self._dist == 0 or self._frame == 0:
            # assume that an empty property means distance has not been calculated
            global PRMTOPPATH

            if self._clusterID == '0_0' and self._ID == 0:  # This is initial, skip all that stuff
                with open(self.path + "/frame_0.pdb") as pdb:
                    self._dist = rmsdDist(pdbLines=list(pdb), refCoords=REFCOORDS, segments=SEGMENTS)
                self.writeInfo()
                return self._dist, 0

            else:  # This is not initial
                def getDist(fr):
                    with open(self.path + "/frame_%i.pdb" % fr) as thisPDB:
                        dist = rmsdDist(pdbLines=list(thisPDB), refCoords=REFCOORDS, segments=SEGMENTS)
                    return fr, dist

                with directory(self.path):
                    sampleFrames = range(int(args.steps / args.sample), args.steps + int(args.steps / args.sample),
                                         int(args.steps / args.sample))
                    distances = parMap(getDist, sampleFrames, n=(cpu_count() / 2))

                    # Get the right frame with min ending distance
                    minDistFrame, minDist = min(distances, key=itemgetter(1))
                    os.rename("frame_%i.rst" % minDistFrame, "end.rst")
                    compress("end.rst")

                    # recreate the coordinate file
                    with open("ptraj.in", 'w') as script:
                        script.write("parm %s\n" % PRMTOPPATH)
                        script.write("trajin coord.nc 1 %i 1\n" % int(minDistFrame / int(args.steps / args.sample)))
                        script.write("trajout coord_new.nc netcdf\n")
                    system("cpptraj < ptraj.in | gzip -f > ptraj.out.gz")
                    os.remove("coord.nc")
                    os.rename("coord_new.nc", "coord.nc")
                    compress("coord.nc")
                    self._dist = minDist
                    self._frame = minDistFrame
                self.writeInfo()
                return self._dist, self._frame
        else:
            return self._dist, self._frame

    def createFolder(self):
        """Creates the top-level folder for this run"""
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def create(self, beginRestart=None, endRestart=None, info=None, coord=None, outFile=None, inFile=None,
               initial=False):
        """Creates the folder structure, writes scripts, and copies compressed input files for a run."""
        global PRMTOPPATH
        self.createFolder()
        with directory(self.path):
            if beginRestart is not None:
                shutil.copy(beginRestart, "begin.rst.gz")
            if endRestart is not None:
                shutil.copy(endRestart, "end.rst.gz")
            if info is not None:  # Copy the run_info and load it
                shutil.copy(info, "run_info")
                self.readInfo()
            else:  # Write a new run_info
                if self._UID == 0 and self._previous != "initial":  # Irresponsible UID assignment
                    self._UID = Run.getNextUID()  # Get a new one
                self.writeInfo()
            self.writeScripts(initial)
            if inFile is not None:
                shutil.copy(inFile, "line.in")
            if coord is not None:
                shutil.copy(coord, "coord.nc.gz")
            if outFile is not None:
                shutil.copy(outFile, "line.out.gz")

    def delete(self, trash=True):
        """Deletes the files associated with this run."""
        global CLUSTERS
        global TRASHPATH
        if self.ID in CLUSTERS[self.clusterID].runs.keys():
            del CLUSTERS[self.clusterID].runs[self.ID]
        if TRASHPATH is not None and trash:
            from time import time

            system("mv -f %s %s/R%s_%i > /dev/null 2> /dev/null" % (self.path, TRASHPATH, self.ID, int(time())))
        else:
            system("rm -rf %s > /dev/null 2> /dev/null" % self.path)  # Works better for some reason

    def writeScripts(self, initial=False):
        """Write the qsub and qscript scripts for this run"""
        global PRMTOPPATH
        with directory(self.path):
            frameSeparation = int(args.steps / args.sample)
            if args.queue_name in ["dept_gpu", "any_gpu", "bahar_gpu"]:
                with open("run.sh", 'w') as runScript:
                    runScript.write("#!/bin/bash\n")
                    runScript.write("sleep 0.5; qsub -d . -q %s -S /bin/bash -N lineMD_R%i -l "
                                    "nodes=1:ppn=1:gpus=1 %s/qscript\n" % (args.queue_name, self._ID, self.path))

                with open("qscript", 'w') as qscript:
                    qscript.write("""#!/bin/bash
AMBERHOME=/usr/local/amber14
PATH=/usr/local/amber14/bin:$PATH
gunzip begin.rst.gz >> out 2>&1
pmemd.cuda -O -i line.in -o line.out -p %s -c begin.rst -r frame -x coord.nc
for f in frame*; do mv "$f" "$f.rst" >> out 2>&1; done
gzip line.out >> out 2>&1
rm mdinfo >> out 2>&1

for i in `seq %i %i %i`;
do
    echo -e "parm %s" >> ptraj_${i}.in
    echo -e "trajin frame_${i}.rst 1 1 1" >> ptraj_${i}.in
    echo -e "trajout frame_${i}.pdb pdb" >> ptraj_${i}.in
    cpptraj < ptraj_${i}.in >> ptraj_frames.out 2>&1
done
gzip ptraj_frames.out >> out 2>&1
touch finished >> out 2>&1
$cmd
""" % (PRMTOPPATH, frameSeparation, frameSeparation, args.steps, PRMTOPPATH))
            elif args.queue_name == "gpu_short":
                with open("run.sh", 'w') as runScript:
                    runScript.write("#!/bin/bash\n")
                    runScript.write("sleep 0.5; qsub -d . -q gpu_short -S /bin/bash -N lineMD_R%i -l "
                                    "nodes=1:ppn=1:gpus=1 "
                                    "-l feature=titan -l walltime=23:59:59 %s/qscript" % (self._ID, self.path))

                with open("qscript", 'w') as qscript:
                    qscript.write("""#!/bin/bash
module purge
module load intel/2013.0
module load amber/14-intel-2013-cuda-5.0
gunzip begin.rst.gz >> out 2>&1
pmemd.cuda -O -i line.in -o line.out -p %s -c begin.rst -r frame -x coord.nc
for f in frame*; do mv "$f" "$f.rst" >> out 2>&1; done
gzip line.out >> out 2>&1
rm mdinfo >> out 2>&1

for i in `seq %i %i %i`;
do
    echo -e "parm %s" >> ptraj_${i}.in
    echo -e "trajin frame_${i}.rst 1 1 1" >> ptraj_${i}.in
    echo -e "trajout frame_${i}.pdb pdb" >> ptraj_${i}.in
    cpptraj < ptraj_${i}.in >> ptraj_frames.out 2>&1
done
gzip ptraj_frames.out >> out 2>&1
touch finished >> out 2>&1
$cmd
""" % (PRMTOPPATH, frameSeparation, frameSeparation, args.steps, PRMTOPPATH))

            os.chmod("run.sh", os.stat("run.sh").st_mode | S_IEXEC)

            with open("line.in", 'w') as inputFile:
                if initial:
                    ig = int(random.random() * 1000.0 % 999)  # For the initial, set the random seed and never rerun it
                    inputFile.write("""&cntrl
 imin = 0, ntx = 1, irest = 0,
 ntpr = 10000, ntwr = -%i, ntwx = %i, ntxo = 1,
 ntf = 2, ntc = 2, cut = 8.0,
 ntb = 2,  nstlim = %i, dt = 0.002,
 temp0 = 300.0, ntt = 3, ig = %i,
 gamma_ln = 1, ioutfm = 1,
 ntp = 1, pres0 = 1.0, taup = 5.0,
/
""" % (int(args.steps / args.sample), int(args.steps / args.sample), args.steps, ig))  # sample, not frame here
                else:  # Restart file
                    ig = self._UID % 999999  # the random seed is UID; should be preserved across moves but still random
                    inputFile.write("""&cntrl
 imin = 0, ntx = 5, irest = 1,
 ntpr = 10000, ntwr = -%i, ntwx = %i, ntxo = 1,
 ntf = 2, ntc = 2, cut = 8.0,
 ntb = 2,  nstlim = %i, dt = 0.002,
 temp0 = 300.0, ntt = 3, ig = %i,
 gamma_ln = 1, ioutfm = 1,
 ntp = 1, pres0 = 1.0, taup = 5.0,
/
""" % (int(args.steps / args.sample), int(args.steps / args.sample), args.steps, ig))

    def writeInfo(self):
        """Write a run_info file for a run."""
        with open(self.path + "/run_info", 'w') as runInfo:
            runInfo.write("""UniqueID: %i
Dist: %.*f
PreviousUID: %s
Explored: %i
Frame: %i
""" % (self._UID, args.precision, self._dist, self._previous, self._explored, self._frame))

    def readInfo(self):
        """Read a run_info file for a run."""
        try:
            info = []
            with open(self.path + "/run_info") as runInfo:
                for line in runInfo:
                    info.append(line.split()[1])
                self._UID = int(info[0])
                self._dist = float(info[1])
                self._previous = info[2]
                self._explored = int(info[3])
                self._frame = int(info[4])
        except IOError:
            log(YELLOW + UNDERLINE +
                "Warning:" + END + YELLOW + " run_info is corrupt or missing; this run will be rejected.\n" + END)
            self.delete()


class Cluster(object):
    """A Cluster object holds runs in a dictionary with ID keys. It knows its innermost distance and can calculate
    the number of runs it holds. The folder format is "C*_*" where "*_*" is the ID or 'R' for the running folder.
    The folder holds a "cluster_info" file to store the distance in persistence, which can be
    manipulated by the writeInfo() and readInfo() methods. The special running cluster should not specify the
    distance and does not support the info file."""

    _ID = '0_0'  # chronologically assigned ID string, also part of name of folder, or may be 'R'
    _runs = None  # dictionary of Run IDs and objects
    _dist = 0  # innermost distance this Cluster holds
    _explored = 0  # cached value for times explored

    def __init__(self, ID, runs=None, dist=0.0, explored=0):
        self._ID = str(ID)
        if runs is None:
            self._runs = {}  # A hack since Python complains about mutable default parameters
        else:
            self._runs = runs
        self._dist = float(dist)
        self._explored = int(explored)

    def __str__(self):
        return "<Cluster: ID %s, %i runs, distance %.*f>" % \
               (self._ID, self.count, args.precision, self._dist)

    @property
    def count(self):
        return len(self.runs)

    @property
    def dist(self):
        return round(self._dist, args.precision)

    @dist.setter
    def dist(self, dist):
        self._dist = round(dist, args.precision)

    @property
    def path(self):  # directory holding runs and the "cluster_info" file
        return "%s/C%s" % (WORKDIR, self._ID)

    @property
    def shortPath(self):
        return "C" + str(self._ID)

    @property
    def ID(self):
        return self._ID

    @property
    def majorID(self):  # part of ID before '_'
        if self._ID == 'R':
            return 'R'
        return int(self._ID.split('_')[0])

    @property
    def minorID(self):  # part of ID after '_'
        if self._ID == 'R':
            return 'R'
        return int(self._ID.split('_')[1])

    @property
    def rawID(self):  # unformatted ID, simply a number
        return int(self.majorID * (int(pow(2, SPLIT))) + self.minorID)

    @property
    def explored(self):
        return self._explored

    @property
    def runs(self):
        return self._runs

    def addRun(self, ID, run):
        """Add a Run to the runs dictionary using ID. Will do nothing if ID already exists."""
        if ID not in self._runs:
            self._runs[ID] = run

    def removeRun(self, ID):
        """Remove the Run with ID ID from the runs dictionary. Returns None if ID is not
        in runs. Returns the run previously there."""
        if ID in self._runs:
            temp = self._runs[ID]
            del self._runs[ID]
            return temp
        return None

    def getRun(self, ID):
        """Gets the run at ID. Returns None if ID is not in runs."""
        if ID in self._runs:
            return self._runs[ID]
        return None

    def setRun(self, ID, run):
        """Adds run to runs at position ID. Returns the run previously there or None."""
        if ID in self._runs:
            temp = self._runs[ID]
            self._runs[ID] = run
            return temp
        return None

    def readRuns(self):
        """Loads the runs in folders into the runs dictionary."""
        if self._ID == 'R':
            return
        self._runs = {}
        runDirectories = [int(name[1:]) for name in os.listdir(self.path) if
                          os.path.isdir(os.path.join(self.path, name))]
        for runID in runDirectories:
            run = Run(ID=runID, clusterID=self._ID)  # No UID because we will readInfo()
            if not run.check("end.rst.gz"):
                if run.check("end.rst"):
                    compress(run.path + "/end.rst")
                else:
                    log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW + " run at " + MAGENTA + str(run.shortPath) +
                        YELLOW + " did not have valid restart file and will be deleted.\n" + END)
                    run.delete()
                continue
            run.readInfo()
            self._runs[runID] = run

    def readExplored(self, exploredDist, start=0):
        """Read the explored count, given a descending-sorted exploredDist based on the explored_dist file.
        Returns an integer that can be used as start to shorten the array loop."""
        exploredCount = 0
        rightBound = self.dist + BINWIDTH
        for index, dist in enumerate(exploredDist[start:]):
            if dist < self.dist:  # We have passed this bin
                self._explored = exploredCount
                return index
            if dist <= rightBound:  # It is inside this bin, counting the right bound
                exploredCount += 1
        self._explored = exploredCount
        return 0  # failsafe

    def adjustedExplored(self, maxBinID):
        """Given the maximum bin with runs, compute the adjusted explored count for this bin."""
        if self.count == 0 or self.explored == 0 or maxBinID <= 0:
            return self.explored
        # favor further bins
        binBias = 2 * (maxBinID - self.rawID)

        # favor bins with more runs
        if self.count == 0:
            countBias = 0
        else:
            countBias = -int(3 * math.log(self.count))
        return self.explored + binBias + countBias

    def writeInfo(self):
        """Write a cluster_info file for a cluster."""
        if self._ID == 'R':
            return
        with open(self.path + "/cluster_info", 'w') as clusterInfo:
            clusterInfo.write("""MinDistance: %.*f
""" % (args.precision, self._dist))

    def readInfo(self):
        """Read a cluster_info file for a cluster."""
        if self._ID == 'R':
            return
        try:
            info = []
            with open(self.path + "/cluster_info") as clusterInfo:
                for line in clusterInfo:
                    info.append(line.split()[1])
                self._dist = float(info[0])
        except IOError:
            log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW +
                " cluster_info corrupt or missing. Will try to rewrite now.\n" + END)
            self.writeInfo()

    def create(self):
        """Create the basic folder structure of a cluster."""
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.writeInfo()

    def explore(self, dist):
        """Mark the distance in dist as explored in the central file and increment the internal count."""
        try:
            with open(WORKDIR + "/explored_dist", 'a') as exploredDist:
                exploredDist.write("%.*f\n" % (args.precision, dist))
        except IOError:
            # Write a new file
            with open(WORKDIR + "/explored_dist", 'w') as exploredDist:
                exploredDist.write("%.*f\n" % (args.precision, dist))
        if self._ID != 'R':
            self._explored += 1


def init():
    """Prepares the initial cluster by copying files. Also begins initial runs based on the thread parameter."""
    global CLUSTERS
    global WORKDIR
    global PRMTOPPATH
    global COORDPATH
    global THREADS

    log(CYAN + "Beginning initialization.\n" + END)
    log("Bin width is " + MAGENTA + "%.*f" % (args.precision, BINWIDTH) + END + ".\n")
    log("Copying and writing input files.\n")

    # Register cluster 0_0 with the dictionary
    CLUSTERS['0_0'] = Cluster(ID='0_0', runs={})  # do not know distance yet
    CLUSTERS['0_0'].create()

    initRun = Run(previous="initial", UID=Run.getNextUID())
    CLUSTERS['0_0'].addRun(ID=0, run=initRun)
    initRun.create(endRestart=COORDPATH, initial=True)
    if not initRun.check("end.rst.gz"):
        fail(RED + UNDERLINE + "Error: " + END + RED + " initial coordinate file is invalid.\n" + END)

    if not os.path.isfile(WORKDIR + "/C0_0/R0/frame_0.pdb"):
        with directory(WORKDIR + "/C0_0/R0"):
            with open("ptraj.in", 'w') as script:
                decompress("end.rst.gz")
                script.write("""parm %s
trajin end.rst 1 1 1
trajout frame_0.pdb pdb
""" % PRMTOPPATH)
            system("cpptraj < ptraj.in > /dev/null 2> /dev/null")
            compress("end.rst")

    # Calculate initial distance
    calcInitDist()

    # Register the running cluster
    CLUSTERS['R'] = Cluster(ID='R', runs={})
    CLUSTERS['R'].create()

    for i in xrange(THREADS):
        # Create runs in the running cluster
        thisRun = Run(ID=i, clusterID='R', previous=initRun.UID, UID=Run.getNextUID())
        CLUSTERS['R'].addRun(ID=i, run=thisRun)
        thisRun.create(beginRestart=COORDPATH, initial=True)
        initRun.explored += 1
        initRun.writeInfo()
        thisRun.execute()  # Begin the run
        # Write to the explored_dist file
        CLUSTERS['R'].explore(CLUSTERS['0_0'].dist + BINWIDTH)
    log(GREEN + "%i initial runs have begun on %s.\n" % (RUNNING, strftime("%c")) + END)


def calcRefCoords():
    """Reads the coordinates of the reference file into the global variable."""
    global REFCOORDS
    global SEGMENTS
    pdbLines = []
    with open(WORKDIR + "/reference.pdb") as pdb:
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


def calcInitDist():
    """Calculates the "initial distance" based on a given run. Sets the variable initDist based on either
    the result of dist() on C0/R0. Also sets the endpoint of the initial bin."""
    global RUNANALYSIS
    if not RUNANALYSIS:
        log("Calculating initial distance.\n")
        initDist = CLUSTERS['0_0'].getRun(ID=0).processDist()[0]
        CLUSTERS['0_0'].dist = initDist - BINWIDTH  # This bin ends here
        CLUSTERS['0_0'].writeInfo()
    else:  # analysis
        initDist = CLUSTERS['0_0'].dist + BINWIDTH
    if not RUNANALYSIS:
        log(BLUE + "Initial RMSD is " + MAGENTA + "%.*f" % (args.precision, initDist) + BLUE +
            " and the first bin ends at " + MAGENTA + "%.*f" % (args.precision,
                                                                CLUSTERS['0_0'].dist) + BLUE + ".\n" + END)
        if args.min is not None:
            log(BLUE + "Analysis RMSD endpoint will be " + MAGENTA + "%.*f" %
                (args.precision, args.min) + BLUE + " angstroms.\n" + END)
            log(BLUE + "Runs ending after " + MAGENTA + "%.*f" % (args.precision, initDist + args.max) +
                BLUE + " angstroms will be rejected.\n")
        else:
            log('\n')


def analysis(runDirectories):
    """Calls the various analysis methods runs with IDs in runDirectories and in the running cluster,
    and decides whether further analysis is necessary."""
    global CLUSTERS
    global WORKDIR
    global SPLIT
    global BINWIDTH
    global RUNANALYSIS
    RUNANALYSIS = True

    log(CYAN + "Beginning analysis.".ljust(getTerminalWidth()) + '\n' + END)

    log("Reading cluster information.\n")
    readClusterInfo(silent=True, readInfo=False, readRuns=False, readExplored=False)  # Read nothing
    determineSplit()

    if not runDirectories:
        readClusterInfo()  # Read everything and print
    else:
        readClusterInfo(silent=True, readExplored=False)  # do not read explored right now
        migrateRuns(runDirectories, oldCluster=CLUSTERS['R'])
        log("Rereading cluster information.\n")
        readClusterInfo()  # Re-read everything after migration and print

    # Signal completion if end cluster reached
    maxCluster = int(max([c.rawID for c in CLUSTERS.values() if c.ID != 'R' and c.count > 0]))
    # Calculate initial and ending distance
    calcInitDist()

    desiredCluster = round((CLUSTERS["0_0"].dist + BINWIDTH - args.min) / BINWIDTH)
    percentage = float(maxCluster) / desiredCluster * 100
    if percentage > 100.0:
        percentage = 100.0
    log(GREEN + "Completed cluster %i of %i total (%.1f%%).\n" % (maxCluster, desiredCluster,
                                                                  percentage) + END)
    if maxCluster < desiredCluster or args.split:
        findNewRuns()
    else:
        stitchTrajectory()
        log(GREEN + "Endpoint reached.\n" + END)
        sys.exit(0)  # End the program now


def readClusterInfo(silent=False, readInfo=True, readRuns=True, readExplored=True):
    """Populate the CLUSTERS database and instruct each cluster to readInfo().
    Note: components of this method are optional for speed.
    silent controls logging, correct is currently disabled, readInfo controls reading distance,
    readRuns controls getting run counts and lists, and readExplored (slowest) controls getting explored count.
    All are True by default. If all are off, the cluster will only know its ID."""
    global CLUSTERS
    CLUSTERS = {}
    clusterDirectories = sorted([name[1:] for name in glob("C*_*") if os.path.isdir(os.path.join(WORKDIR, name))],
                                key=lambda i: (int(i.split('_')[0]), int(i.split('_')[1])))
    # Generate Clusters, read info files, and populate the master dictionary
    # lastDist = None
    # lastID = None
    exploredDist = []
    if readExplored:
        with open(WORKDIR + "/explored_dist") as exploredDistFile:
            for line in exploredDistFile:
                exploredDist.append(float(line))
        exploredDist.sort(reverse=True)
        with open(WORKDIR + "/explored_dist", 'w') as exploredDistFile:
            for dist in exploredDist:
                exploredDistFile.write("%.*f\n" % (args.precision, dist))
    lastStart = 0
    for index, clusterID in enumerate(clusterDirectories):
        cluster = Cluster(ID=clusterID, runs={})  # Do not know dist yet
        CLUSTERS[clusterID] = cluster
        if readInfo:
            cluster.readInfo()  # Now it should have dist
        if readRuns:
            cluster.readRuns()  # Now it has runs and count
        if readExplored:
            lastStart = cluster.readExplored(exploredDist, start=lastStart)  # Now it has explored
        desiredCluster = len(clusterDirectories)
        percentage = float(index) / desiredCluster * 100
        if percentage > 100.0:
            percentage = 100.0
        log("\rReading clusters: %.1f%% complete." % percentage)
    log("\n")
    if not silent:  # print cluster information
        if readInfo:
            maxBinID = 0
            if args.adjust:
                maxBinID = max([c.rawID for c in CLUSTERS.values() if c.count > 0])
            for cluster in sorted(CLUSTERS.values(), key=lambda cl: (cl.majorID, cl.minorID)):
                log("Cluster " + MAGENTA + str(cluster.ID) + END + " from " + MAGENTA + "%.*f"
                    % (args.precision, cluster.dist) + END + " to " + MAGENTA + "%.*f" %
                    (args.precision, cluster.dist + BINWIDTH) + END)
                if readRuns:
                    log(", with " + MAGENTA + str(cluster.count) + END + " runs")
                    if readExplored:
                        log(", explored " + MAGENTA + str(cluster.explored) + END + " times")
                        if args.adjust:
                            log(", adjusted " + MAGENTA + str(cluster.adjustedExplored(maxBinID)) + END + " times.\n")
                        else:
                            log(".\n")
                    else:
                        log(".\n")
                else:
                    log(".\n")

    CLUSTERS['R'] = Cluster(ID='R', runs={})  # Running cluster


def migrateRuns(runDirectories, oldCluster):
    """Examines the runs with IDs in runDirectories within the provided cluster.
     Moves folders to appropriate clusters and creates new ones (if the cluster is the running cluster)."""
    global CLUSTERS
    global NOPROGRESS
    global SPLIT
    if oldCluster.ID != 'R':
        log(BLUE + "Examining cluster " + MAGENTA + "%s.\n" % oldCluster.ID + END)
    else:
        log(BLUE + "Examining the running cluster.\n" + END)
    # Sort the clusters by dist
    sortedClustersList = sorted(CLUSTERS.values(), key=attrgetter("dist"))
    sortedIDsList = [c.ID for c in sortedClustersList]
    for runID in runDirectories:  # should already be sorted
        run = Run(ID=runID, clusterID=oldCluster.ID)  # No UID because we will readInfo()
        if oldCluster.ID == 'R':  # expect a decompressed file
            outFiles = glob(run.path + "/lineMD_R*.o*")
            if outFiles:
                with open(outFiles[0]) as output:
                    out = output.read()
                    if "Calculation halted" in out or "unspecified launch failure" in out \
                            or "busy or unavailable" in out or "STOP PMEMD Terminated Abnormally" in out:
                        log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW + " new run at " + MAGENTA +
                            str(run.shortPath) + YELLOW + " has failed and will be deleted.\n" + END)
                        run.delete()
                        continue
            if not run.check("coord.nc"):  # run verification has failed
                log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW + " new run at " + MAGENTA + str(run.shortPath) +
                    YELLOW + " did not have valid coordinate file and will be deleted.\n" + END)
                run.delete()
                continue
        # otherwise expect compressed file (except initial run); will not be necessary during processDist anyway
        elif not (run.clusterID == '0_0' and run.ID == 0) and not run.check("coord.nc.gz"):  # run verification failed
            log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW + " run at " + MAGENTA + str(run.shortPath) +
                YELLOW + " did not have valid coordinate file and will be deleted.\n" + END)
            run.delete()
            continue
        # run verification succeeded; read distance
        run.readInfo()
        log("Run ID " + MAGENTA + str(run.ID) + END + " (UID " + MAGENTA + str(run.UID) + END + ")")
        dist, frame = run.processDist()
        if dist is None:  # dist returns None if calculation failed
            log(YELLOW + " has failed unexpectedly and will be deleted.\n" + END)
            run.delete()
            continue
        dist = round(float(dist), args.precision)
        log(" has distance " + MAGENTA + "%.*f" % (args.precision, dist) + END + " at frame " + MAGENTA +
            str(frame) + END)
        # Find the right place for this one, if it is in an existing cluster
        found = False
        for cluster in sortedClustersList:
            if cluster.dist < dist <= cluster.dist + BINWIDTH:  # It is inside this bin, counting the left bound
                if cluster.count == 0:  # Successfully moved into a "new" empty bin
                    NOPROGRESS = 0
                newRun = Run.move(run, cluster)
                if run.clusterID != cluster.ID:
                    if cluster.count == 1:  # just populated it
                        log(" and belongs (in new cluster) at " + MAGENTA + "%s.\n" % newRun.shortPath + END)
                    else:
                        log(" and belongs in " + MAGENTA + "%s.\n" % newRun.shortPath + END)
                else:
                    log(" and will stay in the same cluster.\n")
                found = True
                break  # go to next directory
        # Otherwise, make a new cluster if possible
        if not found:
            if oldCluster.ID != 'R':
                log(".\n")
                log(YELLOW + UNDERLINE + "Warning:" + END + YELLOW +
                    " a run does not fit inside new split bins and will be deleted.\n" + END)
                run.delete()
                continue
            # Make a new bin for this one
            initRight = CLUSTERS['0_0'].dist + BINWIDTH
            diff = abs(initRight - dist)
            newRawID = int(math.ceil(diff / float(BINWIDTH)))  # simple number of bins, assuming no split
            if dist > initRight:  # Right of initial cluster
                if dist > initRight + args.max:  # Right of cutoff
                    log(" and is out of cluster range.\n")
                    run.delete()
                    continue
                newMajor = -int(math.ceil(newRawID / pow(2, SPLIT)))
                newMinor = int(math.floor((abs(newMajor) * pow(2, SPLIT) * BINWIDTH - diff) / float(BINWIDTH)))
                newRawID *= -1
            else:
                newRawID -= 1  # There is an off-by-one situation here
                newMajor = int(math.floor(newRawID / pow(2, SPLIT)))
                newMinor = int(newRawID % int(pow(2, SPLIT)))
            newID = "%i_%i" % (newMajor, newMinor)
            if newID in sortedIDsList:
                log(".\n")
                fail(RED + UNDERLINE + "Error:" + END + RED + " run was not found but should be in clusters.\n" + END)
            newDist = CLUSTERS['0_0'].dist - newRawID * BINWIDTH
            cluster = Cluster(ID=newID, runs={}, dist=newDist)
            CLUSTERS[newID] = cluster
            cluster.create()
            newRun = Run.move(run, cluster)
            log(" and belongs (in new cluster) at " + MAGENTA + "%s.\n" % newRun.shortPath + END)
            # Create supplementary split folders
            if SPLIT > 0:
                for thisMinor in xrange(int(pow(2, SPLIT))):
                    thisID = "%i_%i" % (newMajor, thisMinor)
                    if thisID not in sortedIDsList and thisID != newID:  # Is new and not the one we just made
                        thisRawID = newMajor * int(pow(2, SPLIT)) + thisMinor
                        thisDist = round(CLUSTERS['0_0'].dist - thisRawID * BINWIDTH, args.precision)
                        thisCluster = Cluster(ID=thisID, runs={}, dist=thisDist)
                        CLUSTERS[thisID] = thisCluster
                        thisCluster.create()
            if newDist < 0:  # We made a new positive bin
                NOPROGRESS = 0  # We have made progress
            # Re-read to prepare for next iteration through the loop, since CLUSTERS should be accurate now
            sortedClustersList = sorted(CLUSTERS.values(), key=attrgetter("dist"))
            sortedIDsList = [c.ID for c in sortedClustersList]


def findNewRuns():
    """Find the appropriate cluster from which to begin new runs, and prepare and execute runs"""
    global NOPROGRESS
    global NOPROGRESSCUTOFF
    global SPLIT
    global SPLITMAX

    if args.split and SPLIT < SPLITMAX:
        # Split the bins in two, then force analysis to restart. Abandon this attempt to find new runs.
        splitBins()
        finishedRuns = getFinishedRuns()
        if finishedRuns:
            analysis(finishedRuns)
        return

    alreadySelected = False
    # Find cluster with least explored value, then largest bin number (bin will never tie)
    clusterList = [c for c in CLUSTERS.values() if c.ID != 'R' and c.count > 0]
    maxBinID = max([c.rawID for c in clusterList if c.count > 0])
    while RUNNING < THREADS:
        if args.adjust:
            selCluster = min(clusterList, key=lambda cl: (cl.adjustedExplored(maxBinID), cl.dist))
            log("Selected cluster " + MAGENTA + str(selCluster.ID) + END +
                " with " + MAGENTA + str(selCluster.count) + END + " runs, explored " +
                str(selCluster.explored) + " times, adjusted " + MAGENTA +
                str(selCluster.adjustedExplored(maxBinID)) + END + " times.\n")

        else:
            selCluster = min(clusterList, key=lambda cl: (cl.explored, cl.dist))
            log("Selected cluster " + MAGENTA + str(selCluster.ID) + END +
                " with " + MAGENTA + str(selCluster.count) + END + " runs, explored " + MAGENTA +
                str(selCluster.explored) + END + " times.\n")

        if selCluster.ID == '0_0' and SPLIT <= SPLITMAX and not alreadySelected:  # returned to the original bin
            NOPROGRESS += 1
            alreadySelected = True
            log("No progress has been made for " + MAGENTA + str(NOPROGRESS) + END + " of " + MAGENTA +
                str(NOPROGRESSCUTOFF) + END + " iterations.\n")

        # If we have returned to the original cluster too many times:
        if NOPROGRESS >= NOPROGRESSCUTOFF and SPLIT < SPLITMAX:
            # Split the bins in two, then force analysis to restart. Abandon this attempt to find new runs.
            splitBins()
            finishedRuns = getFinishedRuns()
            if finishedRuns:
                analysis(finishedRuns)
            return

        # Pick the least explored run in this cluster. If there is a tie, choose randomly
        runList = selCluster.runs.values()

        def run_cmp(runA, runB):
            if runA.explored > runB.explored:
                return 1
            if runA.explored < runB.explored:
                return -1
            else:
                return random.choice([-1, 1])  # Not 0 because we do not want a tie in this case; non-stable sort

        selRun = min(runList, key=cmp_to_key(run_cmp))  # Sort by explored
        selRun.readInfo()
        log("Selected run " + MAGENTA + "C%s/R%i" % (selCluster.ID, selRun.ID) + END + " (UID " + MAGENTA +
            str(selRun.UID) + END + "), explored " + MAGENTA + str(selRun.explored) + END + " times.\n")

        # Go to running, get the name of the next available run
        runningDirectories = [int(name[1:]) for name in os.listdir(WORKDIR + "/CR") if
                              os.path.isdir(os.path.join(WORKDIR + "/CR", name))]
        if not runningDirectories:
            nextNum = 0
        else:
            nextNum = max(runningDirectories) + 1

        newRun = Run(ID=nextNum, clusterID='R', previous=selRun.UID, UID=Run.getNextUID())
        newRun.create(beginRestart=selRun.path + "/end.rst.gz")
        selRun.explored += 1
        selRun.writeInfo()
        newRun.execute()
        # Write to the explored_dist file
        startDist = selRun.processDist()[0]
        selCluster.explore(startDist)
        log(GREEN + "Executed run %i of %i (new ID %i, unique ID %i, start distance %.*f) on %s.\n" %
            (RUNNING, THREADS, newRun.ID, newRun.UID, args.precision, startDist, strftime("%c")) + END)


def splitBins():
    """Perform bin splitting and run migration to new clusters"""
    global SPLIT
    global BINWIDTH
    global CLUSTERS
    global NOPROGRESS

    SPLIT += 1
    BINWIDTH = round(BINWIDTH / 2.0, args.precision)

    log(CYAN + "Splitting bins for iteration %i of %i.\n" % (SPLIT, SPLITMAX) + END)

    # delete clusters beyond range
    clustersList = copy.copy(CLUSTERS.values())
    for cluster in clustersList:
        if cluster.dist >= CLUSTERS['0_0'].dist + BINWIDTH + args.max and cluster.ID != 'R':  # Found one bad cluster
            majorID = cluster.majorID
            for otherCluster in clustersList:
                if otherCluster.majorID == majorID and otherCluster.ID in CLUSTERS:
                    # Don't care about the minor ID. Just delete the whole set
                    shutil.rmtree(otherCluster.path)
                    del CLUSTERS[otherCluster.ID]

    # find available IDs
    nextMinor = max([c.minorID for c in CLUSTERS.values() if c.ID != 'R']) + 1

    # create new clusters in database
    # The clusters before split:
    oldClusterList = sorted([c for c in CLUSTERS.values() if c.ID != 'R'], key=attrgetter("dist"))
    zeroClusterList = [c for c in oldClusterList if c.minorID == 0]  # Only clusters "C*_0"
    for zeroCluster in zeroClusterList:
        # Create new clusters
        for i in xrange(int(pow(2, SPLIT - 1))):  # Repeat for each new folder, based on the number of previous splits
            newID = "%i_%i" % (zeroCluster.majorID, nextMinor + i)
            newCluster = Cluster(ID=newID, runs={}, dist=0)
            CLUSTERS[newID] = newCluster
            # create new folders and cluster info
            newCluster.create()
            newCluster.writeInfo()

        # change cluster info
        rightBound = zeroCluster.dist + BINWIDTH * 2  # BINWIDTH has changed
        for i in xrange(int(pow(2, SPLIT))):  # Repeat for each existing folder
            ID = "%i_%i" % (zeroCluster.majorID, i)  # This includes zeroCluster
            cluster = CLUSTERS[ID]
            # Set the distance to the old cluster's right bound, then move it to the left
            cluster.dist = round(rightBound - BINWIDTH * (i + 1), args.precision)
            cluster.writeInfo()

    # re-read cluster info into database to prepare for move
    log("Reading new cluster information.\n")
    readClusterInfo(silent=True, readExplored=False)  # explored count is not necessary

    # move runs for each old cluster; new clusters should not be necessary now.
    for cluster in oldClusterList:
        migrateRuns([run.ID for run in cluster.runs.values()], cluster)

    log("Rereading cluster information.\n")
    readClusterInfo()  # read everything

    log("Finished run migration.\n")
    NOPROGRESS = 0
    return  # Allow findNewRuns to take back to analysis


if __name__ == "__main__":
    main()
