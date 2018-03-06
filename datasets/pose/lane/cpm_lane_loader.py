#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Class for the Lane Pose Loader.


from __future__ import absolute_import
from __future__ import division

import json
import os

import torch.utils.data as data
from PIL import Image

from datasets.pose.pose_data_utilizer import PoseDataUtilizer
from utils.logger import Logger as Log


class CPMLaneLoader(data.Dataset):

    def __init__(self, root_dir, base_transform=None,
                 input_transform=None, heatmap_transform=None,
                 split="train", configer=None):
        (self.img_list, self.json_list) = self.__list_dirs(root_dir)
        self.split = split
        self.configer = configer
        self.base_transform = base_transform
        self.input_transform = input_transform
        self.heatmap_transform = heatmap_transform
        self.pose_utilizer = PoseDataUtilizer(configer)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        kpts = self.__get_kpts(self.json_list[index])

        mask = None
        center = None
        if self.base_transform is not None:
            img, _, mask, kpts, center = self.base_transform(img, mask=mask, kpt=kpts, center=center)

        heatmap = self.pose_utilizer.generate_heatmap(kpts=kpts, mask=mask)

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.heatmap_transform is not None:
            heatmap = self.heatmap_transform(heatmap)

        return img, heatmap

    def __len__(self):

        return len(self.img_list)

    def __get_kpts(self, json_path):
        kpt = list()
        fp = open(json_path)
        node = json.load(fp)
        for item in node['objects']:
            # if 'arrow' not in item['label'].split('_'):
            #    continue
            # if item['label'] != 'arrow_straight':
            #    continue

            for points in item['polygon']:
                ind = list()
                ind.append(list([points[0], points[1], 1]))
                kpt.append(ind)

        return kpt

    def __list_dirs(self, root_dir):
        img_list = list()
        json_list = list()
        image_dir = os.path.join(root_dir, 'image')
        json_dir = os.path.join(root_dir, 'json')
        for file_name in os.listdir(image_dir):
            image_name = file_name.split('.')[0]
            img_list.append(os.path.join(image_dir, file_name))
            json_path = os.path.join(json_dir, '{}.json'.format(image_name))
            json_list.append(json_path)
            if not os.path.exists(json_path):
                Log.error('Json Path: {} not exists.'.format(json_path))
                exit(0)

        return img_list, json_list


if __name__ == "__main__":
    # Test Lane Loader.
    pass