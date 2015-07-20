#!/bin/env python
# histogram.py
# Generate histogram from a list of data points

from shared import *


def histogram(dataList, binWidth, normalize=False):
    """Create a set of bins and frequencies from a list of data.
    Arguments:
        dataList: list of float values for histogram
        normalize: if True, makes bin frequencies normalized to 1 as maximum
    Returns:
        list of tuples with first item being the maximum value inside a bin and
        the second being the frequency
    """

    # dataList: [value1, value2, ...]
    dataCount = len(dataList)

    dataList.sort()
    dataList.reverse()

    bottom = math.floor(float(min([float(x) for x in dataList])) / binWidth) * binWidth
    top = math.ceil(float(max([float(x) for x in dataList])) / binWidth) * binWidth
    binCounts = []
    # binCounts: [(binTop1, count1), (binTop2, count2), ...]
    # Start from beginning of file
    currentBinTop = top
    currentListIndex = 0
    currentBinCount = 0

    while currentBinTop > bottom and currentListIndex < len(dataList):
        # Count how many are in this bin
        if (float(dataList[currentListIndex]) <= currentBinTop and float(
                dataList[currentListIndex]) > currentBinTop - binWidth):
            currentBinCount += 1
            currentListIndex += 1
        else:
            binCounts.append((currentBinTop, currentBinCount))
            currentBinTop -= binWidth
            currentBinCount = 0

    for index in xrange(len(binCounts)):
        if not normalize:
            binCounts[index] = (binCounts[index][0], float(binCounts[index][1]))
        else:
            binCounts[index] = (binCounts[index][0], float(binCounts[index][1]) / dataCount)

    return binCounts


def main():
    parse()
    # Input data
    # inputList: [value1, value2, ...]
    inputList = []
    with open(args.file, 'r') as theFile:
        for line in theFile.readlines():
                # Import data into table
            if not line.startswith("#"):
                try:
                    inputList.append(float(line.split(" ")[args.column - 1]))
                except ValueError:
                    pass
    for item in histogram(inputList, args.bin):
        print str(item[0]) + " " + str(item[1])


def parse():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Generate histogram from a list of data points")
    parser.add_argument("file", help="list of data points", type=str)
    parser.add_argument('-b', "--bin", help="width of bin in angstroms", type=float, default=0.5)
    parser.add_argument('-c', "--column", help="column containing data", type=int, default=1)
    global args
    args = parser.parse_args()

if __name__ == "__main__":
    main()
