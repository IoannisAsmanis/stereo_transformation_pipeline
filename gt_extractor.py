#!/usr/bin/env python3

import sys
import os

def readFileToList(fileName, poseStartIdx, poseEndIdx, delim):
    with open(fileName, "r") as f:
        data = f.readlines()
    pose_list = []
    for line in data:
        numbers = [float(x) for x in line.split(delim)]
        pose_list.append(numbers[poseStartIdx:poseEndIdx])
    assert(len(data) == len(pose_list))
    return pose_list


def writeToFile(poses, outputFileName, outputIdxs=None, offsetId=None):
    counter = 0
    outputFilt = outputIdxs if (outputIdxs != None) else list(range(len(poses[0])))
    offsets = [poses[offsetId][of] for of in outputFilt] if (offsetId != None) else [0] * len(outputFilt)
    print(offsets)

    with open(outputFileName, "w") as f:
        for pose in poses:
            f.write(str(counter))
            f.write('\t')
            counter += 1

            for i in range(len(outputFilt)):
                f.write("%f" % (pose[outputFilt[i]] - offsets[i]))
                f.write('\t' if i < len(outputFilt)-1 else '\n')


if __name__=='__main__':
    if (len(sys.argv) != 3):
        print("Wrong number of args supplied!")
        sys.exit(1)

    poses = readFileToList(sys.argv[1], 3, 10, ' ')
    writeToFile(poses, sys.argv[2], list(range(3)), 0)
