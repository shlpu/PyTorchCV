#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Fashion AI data loader for keypoints detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import math
import numpy as np
import torch.utils.data as data
from PIL import Image

from datasets.pose.pose_data_utilizer import PoseDataUtilizer
from utils.logger import Logger as Log


class FAFaiLoader(data.Dataset):

    def __init__(self, root_dir = None, base_transform=None,
                 input_transform=None, label_transform=None,
                 split="train", configer=None):

        self.img_list, self.json_list = self.__list_dirs(root_dir)
        self.split = split
        self.configer = configer
        self.base_transform = base_transform
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.pose_utilizer = PoseDataUtilizer(configer)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')

        kpts, cat = self.__read_json_file(self.json_list[index])

        if self.base_transform is not None:
            img, _, kpts, center = self.base_transform(img, mask=None, kpt=kpts, center=None)

        width, height = img.size
        heat_mask = np.zeros((width // self.configer.get('network', 'stride'),
                            height // self.configer.get('network', 'stride'),
                            self.configer.get('network', 'heatmap_out')), dtype=np.uint8)
        heat_mask[:, :, self.configer.get('mask', cat)] = 1

        heatmap = self.__generate_heatmap(kpts=kpts, mask=heat_mask)

        vec_mask = np.ones((width // self.configer.get('network', 'stride'),
                              height // self.configer.get('network', 'stride'),
                              self.configer.get('network', 'paf_out')), dtype=np.uint8)

        vec_mask = self.__get_vecmask(kpts, cat, vec_mask)

        vecmap = self.pose_utilizer.generate_paf(kpts=kpts, mask=vec_mask)

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.label_transform is not None:
            heatmap = self.label_transform(heatmap)
            vecmap = self.label_transform(vecmap)
            heat_mask = self.label_transform(heat_mask)
            vec_mask = self.label_transform(vec_mask)

        return img, heatmap, heat_mask, vecmap, vec_mask

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

        for object in node['objects']:
            kpts.append(object['keypoints'])

        fp.close()
        return kpts, node['category']

    def __get_vecmask(self, kpts, cat, vec_mask):
        for i in range(len(self.configer.get('details', 'limb_seq'))):
            pair = [x-1 for x in self.configer.get('details', 'limb_seq')[i]]
            if pair[0] not in self.configer.get('mask', cat) or pair[1] not in self.configer.get('mask', cat):
                vec_mask[:, :, [i*2, i*2+1]] = 0

        return vec_mask

    def __generate_heatmap(self, kpts=None, mask=None):

        height = self.configer.get('data', 'input_size')[1]
        width = self.configer.get('data', 'input_size')[0]
        stride = self.configer.get('network', 'stride')
        num_keypoints = self.configer.get('data', 'num_keypoints')
        sigma = self.configer.get('heatmap', 'sigma')
        method = self.configer.get('heatmap', 'method')

        heatmap = np.zeros((height // stride,
                            width // stride,
                            num_keypoints + 1), dtype=np.float32)
        start = stride / 2.0 - 0.5
        num_objects = len(kpts)
        for i in range(num_objects):
            for j in range(num_keypoints):
                if kpts[i][j][2] != 1:
                    continue

                x = kpts[i][j][0]
                y = kpts[i][j][1]
                for h in range(height // stride):
                    for w in range(width // stride):
                        xx = start + w * stride
                        yy = start + h * stride
                        dis = 0
                        if method == 'gaussian':
                            dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                        elif method == 'laplace':
                            dis = math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma
                        else:
                            Log.error('Method: {} is not valid.'.format(method))
                            exit(1)

                        if dis > 4.6052:
                            continue

                        # Use max operator?
                        heatmap[h][w][j] = max(math.exp(-dis), heatmap[h][w][j])
                        if heatmap[h][w][j] > 1:
                            heatmap[h][w][j] = 1

        heatmap[:, :, num_keypoints] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)
        if mask is not None:
            heatmap = heatmap * mask

        return heatmap

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
                exit(1)

        return img_list, json_list


if __name__ == "__main__":
    # Test coco loader.
    pass