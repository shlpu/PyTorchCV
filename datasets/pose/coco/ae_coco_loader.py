#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Coco data loader for keypoints detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import torch.utils.data as data
from PIL import Image

from datasets.pose.pose_data_utilizer import PoseDataUtilizer
from utils.logger import Logger as Log


class AECocoLoader(data.Dataset):

    def __init__(self, root_dir = None, base_transform=None,
                 input_transform=None, label_transform=None,
                 split="train", configer=None):

        self.img_list, self.json_list, self.mask_list = self.__list_dirs(root_dir)
        self.split = split
        self.configer = configer
        self.base_transform = base_transform
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.pose_utilizer = PoseDataUtilizer(configer)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        maskmap = Image.open(self.mask_list[index]).convert('P')

        kpts, center, scale = self.__read_json_file(self.json_list[index])

        if self.base_transform is not None:
            img, maskmap, kpts, center = self.base_transform(img, mask=maskmap, kpt=kpts, center=center)

        width, height = maskmap.size
        maskmap = maskmap.resize((width // self.configer.get('network', 'stride'),
                                  height // self.configer.get('network', 'stride')), Image.NEAREST)

        maskmap = np.expand_dims(np.array(maskmap, dtype=np.float32), axis=2)

        heatmap = self.pose_utilizer.generate_heatmap(kpts=kpts, mask=maskmap)

        tagmap, num_objects = self.pose_utilizer.generate_tagmap(kpts=kpts)

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.label_transform is not None:
            heatmap = self.label_transform(heatmap)
            maskmap = self.label_transform(maskmap)

        return img, heatmap, maskmap, tagmap, num_objects

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, json_file):
        """
            filename: JSON file

            return: three list: key_points list, centers list and scales list.
        """
        fp = open(json_file)
        node = json.load(fp)

        kpts = list()
        centers = list()
        scales = list()

        for object in node['persons']:
            kpts.append(object['keypoints'])
            centers.append(object['pos_center'])
            scales.append(object['scale'])

        fp.close()

        return kpts, centers, scales

    def __list_dirs(self, root_dir):
        img_list = list()
        json_list = list()
        mask_list = list()
        image_dir = os.path.join(root_dir, 'image')
        json_dir = os.path.join(root_dir, 'json')
        mask_dir = os.path.join(root_dir, 'mask')

        for file_name in os.listdir(image_dir):
            image_name = file_name.split('.')[0]
            img_list.append(os.path.join(image_dir, file_name))
            mask_path = os.path.join(mask_dir, '{}.png'.format(image_name))
            mask_list.append(mask_path)
            json_path = os.path.join(json_dir, '{}.json'.format(image_name))
            json_list.append(json_path)
            if not os.path.exists(json_path):
                Log.error('Json Path: {} not exists.'.format(json_path))
                exit(1)

        return img_list, json_list, mask_list


if __name__ == "__main__":
    # Test coco loader.
    pass