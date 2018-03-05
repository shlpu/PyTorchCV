#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualize the tensor of the detection.


import cv2
import os

from utils.logger import Logger as log


BBOX_DIR = 'vis/results/det/bboxes'


class DetVisualizer(object):

    def __init__(self, configer):
        self.configer = configer

    def vis_bboxes(self, image, bboxes_list,
                   name='default', vis_dir=BBOX_DIR,
                   scale_factor=1, img_size=None):
        """
          Show the diff bbox of individuals.
        """
        vis_dir = os.path.join(self.configer.get('project_dir'), vis_dir)

        if not os.path.exists(vis_dir):
            log.error('Dir:{} not exists!'.format(vis_dir))
            os.makedirs(vis_dir)

        img_path = os.path.join(vis_dir, '{}.jpg'.format(name))

        for bbox in bboxes_list:
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),-1)

        image = self.scale_image(image, scale_factor, img_size)
        cv2.imwrite(img_path, image)

    def scale_image(self, image, scale_factor=1,
                    img_size=None, inter_method=cv2.INTER_CUBIC):
        if img_size is not None:
            image = cv2.resize(image, img_size, interpolation=inter_method)
        else:
            image = cv2.resize(image, None, fx=scale_factor,
                               fy=scale_factor, interpolation=inter_method)

        return image