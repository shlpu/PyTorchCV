#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Seg Model for semantic segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.seg.erf_net import ERFNet
from utils.logger import Logger as Log


class SegModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def seg_net(self):
        key = self.configer.get('network', 'model_name')
        if key == 'erf_net':
            return ERFNet(self.configer.get('network', 'out_channels'))
        else:
            Log.error('Model: {} not valid!'.format(key))
            exit(1)