#!/usr/bin/env python3

import sys
import os

"""
Read a file from disk and convert the data to a list of lists of floats
Only keep data per line from poseStartIdx to poseEndIdx
"""
def readFileToList(fileName, poseStartIdx, poseEndIdx, delim):
    with open(fileName, "r") as f:
        data = f.readlines()
    pose_list = []
    for line in data:
        numbers = [float(x) for x in line.split(delim)]
        pose_list.append(numbers[poseStartIdx:poseEndIdx])
    assert(len(data) == len(pose_list))
    return pose_list


"""
Filter through a list of lists of floats
Each line in the output will contain:
1. an incrementing counter per line
2. the pose data
Additionally, if offsetId is NOT None, every pose will become
equal to its value - the value of the pose at offsetId,
meaning that at poses[offsetId] will be treated as 0
"""
def writeToFile(poses, outputFileName, outputIdxs=None, offsetId=None):
    counter = 0
    outputFilt = outputIdxs if (outputIdxs != None) else list(range(len(poses[0])))
    offsets = [poses[offsetId][of] for of in outputFilt] if (offsetId != None) else [0] * len(outputFilt)
    #print(offsets)

    with open(outputFileName, "w") as f:
        for pose in poses:
            f.write(str(counter))
            f.write('\t')
            counter += 1

            for i in range(len(outputFilt)):
                f.write("%f" % (pose[outputFilt[i]] - offsets[i]))
                f.write('\t' if i < len(outputFilt)-1 else '\n')


# Driver code, usage: $ python3 gt_extractor.py <ALL_METADATA_FILE_PATH> <OUTPUT_FILE_PATH>
if __name__=='__main__':
    if (len(sys.argv) != 3):
        print("Wrong number of args supplied!")
        sys.exit(1)

    poses = readFileToList(sys.argv[1], 3, 10, ' ')
    writeToFile(poses, sys.argv[2], list(range(3)), 0)
