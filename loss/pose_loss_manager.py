#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Pose Estimation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.pose_modules import CapsuleLoss, MarginLoss, MseLoss
from loss.modules.pose_modules import EmbeddingLoss
from loss.modules.pose_modules import VoteLoss
from utils.logger import Logger as Log


class PoseLossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_pose_loss(self, key):
        if key == 'mse_loss':
            return MseLoss()

        elif key == 'capsule_loss':
            return CapsuleLoss(num_keypoints=self.configer.get('data', 'num_keypoints'),
                               l_vec=self.configer.get('capsule', 'l_vec'))

        elif key == 'margin_loss':
            return MarginLoss(num_keypoints=self.configer.get('data', 'num_keypoints'),
                              l_vec=self.configer.get('capsule', 'l_vec'))

        else:
            Log.error('Pose Loss: {} is not valid.'.format(key))
            exit(1)

    def get_relation_loss(self, key):
        if key == 'embedding_loss':
            return EmbeddingLoss(num_keypoints=self.configer.get('data', 'num_keypoints'),
                                 l_vec=self.configer.get('capsule', 'l_vec'))

        else:
            Log.error('Relation loss: {} is not valid.'.format(key))
            exit(1)

    def get_vote_loss(self, key):
        if key == 'vote_loss':
            return VoteLoss()

        else:
            Log.error('Vote loss: {} is not valid.'.format(key))
            exit(1)
