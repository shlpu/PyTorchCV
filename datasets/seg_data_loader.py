#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Class for the Semantic Segmentation Data Loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils import data

import datasets.tools.transforms as trans
from utils.logger import Logger as Log


class SegDataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        self.base_train_transform = trans.BaseCompose([
            trans.RandomResize(),
            trans.RandomRotate(self.configer.get('data', 'rotate_degree')),
            trans.RandomCrop(self.configer.get('input_size')),
            trans.RandomResize(size=self.configer.get('input_size')), ])

        self.base_val_transform = trans.BaseCompose([
            trans.RandomResize(size=self.configer.get('input_size')), ])

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=[128.0, 128.0, 128.0],
                            std=[256.0, 256.0, 256.0]), ])

        self.label_transform = trans.Compose([trans.ToTensor(), ])

    def get_trainloader(self, Loader=None):
        if self.configer.get('dataset') == 'cityscape':
            cityscape_trainloader = data.DataLoader(
                Loader(root_dir=self.configer.get('train_dir'),
                       base_transform=self.base_train_transform,
                       img_transform=self.img_transform,
                       label_transform=self.label_transform,
                       split='train'),
                batch_size=self.configer.get('batch_size'), shuffle=True,
                num_workers=self.configer.get('solver', 'workers'), pin_memory=True)

            return cityscape_trainloader

        else:
            Log.error('Dataset: {} is invalid.'.format(self.configer.get('dataset')))
            return None

    def get_valloader(self, Loader=None):
        if self.configer.get('dataset') == 'cityscape':
            cityscape_valloader = data.DataLoader(
                Loader(root_dir=self.configer.get('val_dir'),
                       base_transform=self.base_val_transform,
                       img_transform=self.img_transform,
                       label_transform=self.label_transform,
                       split='val'),
                batch_size=self.configer.get('batch_size'), shuffle=False,
                num_workers=self.configer.get('solver', 'workers'), pin_memory=True)

            return cityscape_valloader

        else:
            Log.error('Dataset: {} is invalid.'.format(self.configer.get('dataset')))
            return None

if __name__ == "__main__":
    # Test data loader.
    pass