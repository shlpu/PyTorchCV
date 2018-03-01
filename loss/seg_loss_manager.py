#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss Manager for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.seg_modules import CrossEntropyLoss, IOULoss, EmbeddingLoss, FocalLoss
from utils.logger import Logger as Log


class SegLossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_seg_loss(self, key):
        if key == 'cross_entropy_loss':
            return CrossEntropyLoss()
        elif key == 'embedding_loss':
            return EmbeddingLoss(self.configer.get('num_classes'))
        elif key == 'iou_loss':
            return IOULoss(self.configer.get('num_classes'))
        elif key == 'focal_loss':
            return FocalLoss(self.configer.get('focal', 'y'))
        else:
            Log.error('Segmentation Loss: {} is not valid.'.format(key))
            exit(1)