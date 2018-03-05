#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Parse json file of coco keypoints.


import json
import cv2
import sys
import math
import matplotlib
import pylab as plt
import numpy as np


part_str = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", 
            "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Leye", "Reye", 
            "Lear", "Rear", "pt19"]

limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], [10,11], [12,13], [13,14],
           [1,2], [2,9], [2,12], [2,3], [2,6], [1,16],[1,15], [16,18], [15,17], [3,17],[6,18]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
          [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
          [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
          [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def parse_json(json_file, image_file):
    canvas = cv2.imread(image_file) # B,G,R order

    with open(json_file) as infile:
        info = json.load(infile)

    pose_list = list()
    count = 1
    for item in info['persons']:
        count += 1
        pose_list.append(item['keypoints'])

    pose_array = np.array(pose_list)
    print pose_array.shape

    cmap = matplotlib.cm.get_cmap('hsv')
    
    for i in range(18):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(pose_array.shape[0]):
            if pose_array[j][i][2] == 2:
                continue

            cv2.circle(canvas, (int(pose_array[j][i][0]), int(pose_array[j][i][1])), 4, colors[i], thickness=-1)

    cv2.imshow("main", canvas)
    cv2.waitKey()

    for i in range(17):
        for n in range(pose_array.shape[0]):
            Y = pose_array[n][np.array(limbSeq[i])-1][:,0]
            X = pose_array[n][np.array(limbSeq[i])-1][:,1]
            if 2. in pose_array[n][np.array(limbSeq[i])-1][:,2]:
                continue

            cur_canvas = canvas.copy()

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    cv2.imshow("main", canvas)
    cv2.waitKey()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Need two args: json_file & image_path"
        exit(0)

    json_file = sys.argv[1]
    image_file = sys.argv[2]
    parse_json(json_file, image_file)
