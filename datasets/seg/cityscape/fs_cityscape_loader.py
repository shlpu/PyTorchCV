#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
from torch.utils import data

from datasets.tools.transforms import ReLabel


class FSCityScapeLoader(data.Dataset):
    def __init__(self, root_dir, split="train", base_transform=None,
                 img_transform=None, label_transform=None):
        self.base_transform = base_transform
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.img_list, self.label_list = self.__list_dirs(root_dir, split)

    def __list_dirs(self, root_dir, split, coarse=False):
        img_list = list()
        label_list = list()
        img_dir = os.path.join(root_dir, 'leftImg8bit', split)
        label_dir = os.path.join(root_dir, 'gtFine', split)

        for sub_dir in os.listdir(img_dir):
            for sub_file in os.listdir(os.path.join(img_dir, sub_dir)):
                filename = '_'.join(sub_file.split('_')[:-1])
                img_list.append(os.path.join(img_dir, sub_dir, '{}_leftImg8bit.png'.format(filename)))
                label_list.append(os.path.join(label_dir, sub_dir,
                                               '{}_gtFine_labelTrainIds.png'.format(filename)))

        if coarse:
            coarse_img_dir = os.path.join(root_dir, "leftImg8bit/train_extra")
            coarse_label_dir = os.path.join(root_dir, "gtCoarse/train_extra")

            for sub_dir in os.listdir(coarse_img_dir):
                for sub_file in os.listdir(os.path.join(coarse_img_dir, sub_dir)):
                    filename = '_'.join(sub_file.split('_')[:-1])
                    img_list.append(os.path.join(coarse_img_dir, sub_dir,
                                                 '{}_leftImg8bit.png'.format(filename)))
                    label_list.append(os.path.join(coarse_label_dir, sub_dir,
                                                   '{}_gtCoarse_labelTrainIds.png'.format(filename)))

        return img_list, label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = self.img_list[index]
        label_file = self.label_list[index]

        img = Image.open(img_file).convert('RGB')
        label = Image.open(label_file).convert("P")

        if self.base_transform is not None:
            img, label = self.base_transform(img, label=label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        label = ReLabel(255, 19)(label)
        return img, label


if __name__ == "__main__":
    # Test cityscapes loader.
    pass
