#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Parse json file of lane keypoints.


import sys
import os
import cv2
import json
import matplotlib
import pylab as plt
import numpy as np


def parse_json(json_file, image_file, show_points=1):
    canvas = cv2.imread(image_file)  # B,G,R order

    with open(json_file) as infile:
        info = json.load(infile)

    pose_list = list()
    pose_list = info['objects']

    cmap = matplotlib.cm.get_cmap('hsv')

    for i in range(len(pose_list)):
        rgba = np.array(cmap(1 - i / 18. - 1. / 36))
        rgba[0:3] *= 255
        count = 1
        max_co = (0, 0)
        for coordinate in (pose_list[i]['polygon']):
            if coordinate[1] > max_co[1]:
                max_co = coordinate
        cv2.circle(canvas, (int(max_co[0]), int(max_co[1])), 4, [0, 255, 0], thickness=-1)
        cv2.imshow("main", canvas)
        cv2.waitKey()
        #count += 1
        #if count > show_points:
        # break

    cv2.imshow("main", canvas)
    cv2.waitKey()


def parse_json_dir(json_dir, image_dir):
    for filename in os.listdir(image_dir):
        image_file = image_dir + '/' + filename
        json_file = json_dir + '/' + filename.split('.')[0] + '.json'
        parse_json(json_file, image_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Need two args: json_file & image_path"
        exit(0)

    json_file = sys.argv[1]
    image_file = sys.argv[2]
    parse_json_dir(json_file, image_file)
