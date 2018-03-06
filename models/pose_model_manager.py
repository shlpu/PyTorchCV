#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Pose Model for pose detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.pose.capsule_net import CapsuleNet
from models.pose.cpm_net import CPMNet
from models.pose.open_pose import get_open_pose
from models.pose.fashion_ai import get_fashion_ai

from models.pose.simple_net import SimpleNet
from utils.logger import Logger as Log


class PoseModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def pose_detector(self):
        key = self.configer.get('network', 'model_name')
        if key == 'cpm_net':
            return CPMNet(self.configer.get('network', 'out_channels'))

        elif key == 'simple_net':
            return SimpleNet(self.configer.get('network', 'out_channels'))

        elif key == 'open_pose':
            return get_open_pose(self.configer.get('network', 'paf_out'),
                                 self.configer.get('network', 'heatmap_out'))

        elif key == 'fashion_ai':
            return get_fashion_ai(self.configer.get('network', 'paf_out'),
                                 self.configer.get('network', 'heatmap_out'))

        else:
            Log.error('Model: {} not valid!'.format(key))
            exit(0)

    def human_detector(self):
        pass

    def human_filter(self):
        return CapsuleNet()