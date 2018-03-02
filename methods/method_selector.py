#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from methods.pose.conv_pose_machine import ConvPoseMachine
from methods.pose.conv_pose_machine_test import ConvPoseMachineTest
from methods.pose.open_pose import OpenPose
from methods.pose.open_pose_test import OpenPoseTest
from methods.pose.associative_embedding import AssociativeEmbedding
from methods.pose.associative_embedding_test import AssociativeEmbeddingTest

from methods.seg.fcn_segmentor import FCNSegmentor
from methods.seg.fcn_segmentor_test import FCNSegmentorTest

from utils.logger import Logger as Log


class MethodSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def select_pose_model(self):
        key = self.configer.get('method')
        if key == 'open_pose':
            if self.configer.get('phase') == 'train':
                return OpenPose(self.configer)
            else:
                return OpenPoseTest(self.configer)

        elif key == 'conv_pose_machine':
            if self.configer.get('phase') == 'train':
                return ConvPoseMachine(self.configer)
            else:
                return ConvPoseMachineTest(self.configer)

        elif key == 'associative_embedding':
            if self.configer.get('phase') == 'train':
                return AssociativeEmbedding(self.configer)
            else:
                return AssociativeEmbeddingTest(self.configer)

        else:
            Log.error('Pose Model: {} is not valid.'.format(key))
            exit(1)

    def select_det_model(self):
        key = self.configer.get('method')
        if key == 'pose_top_down':
            return ConvPoseMachine(self.configer)

        else:
            Log.error('Det Model: {} is not valid.'.format(key))

    def select_seg_model(self):
        key = self.configer.get('method')
        if key == 'fcn_segmentor':
            if self.configer.get('phase') == 'train':
                return FCNSegmentor(self.configer)
            else:
                return FCNSegmentorTest(self.configer)

        else:
            Log.error('Seg Model: {} is not valid.'.format(key))
