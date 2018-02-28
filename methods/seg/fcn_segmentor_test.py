#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
import numpy as np

from torch.autograd import Variable

from datasets.tools.transforms import RandomResize, ToTensor, Normalize
from models.seg_model_manager import SegModelManager
from methods.tools.module_utilizer import ModuleUtilizer
from vis.visualizer.seg_visualizer import SegVisualizer
from utils.logger import Logger as Log


class FCNSegmentorTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.seg_vis = SegVisualizer(configer)
        self.seg_model_manager = SegModelManager(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.seg_net = None

    def init_model(self):
        self.seg_net = self.seg_model_manager.seg_net()
        self.seg_net, _ = self.module_utilizer.load_net(self.seg_net)
        self.seg_net.eval()

    def forward(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = RandomResize(size=self.configer.get('data', 'input_size'))(image)
        image = ToTensor()(image)
        image = Normalize(mean=[128.0, 128.0, 128.0], std=[256.0, 256.0, 256.0])(image)
        inputs = Variable(image.unsqueeze(0).cuda(), volatile=True)
        results = self.seg_net.forward(inputs)
        return results.data.cpu().numpy().argmax(axis=1)[0].squeeze()

    def __test_img(self, image_path, save_path):
        if self.configer.get('dataset') == 'cityscape':
            self.__test_cityscape_img(image_path, save_path)
        elif self.configer.get('dataset') == 'laneline':
            self.__test_laneline_img(image_path, save_path)
        else:
            Log.error('Dataset: {} is not valid.'.format(self.configer.get('dataset')))
            exit(1)

    def __test_cityscape_img(self, img_path, save_path):
        color_list = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                      (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                      (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                      (0, 80, 100), (0, 0, 230), (119, 11, 32)]

        result = self.forward(img_path)
        width = self.configer.get('data', 'input_size')[0] // self.configer.get('network', 'stride')
        height = self.configer.get('data', 'input_size')[1] // self.configer.get('network', 'stride')
        color_dst = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            color_dst[result == i] = color_list[i]

        color_img = np.array(color_dst, dtype=np.uint8)
        color_img = Image.fromarray(color_img, 'RGB')
        color_img.save(save_path)

    def __test_laneline_img(self, img_path, save_path):
        pass

    def test(self, test_img=None, test_dir=None):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/pose', self.configer.get('dataset'), 'test')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            filename = test_img.rstrip().split('/')[-1]
            save_path = os.path.join(base_dir, filename)
            self.__test_img(test_img, save_path)

        else:
            for filename in self.__list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                save_path = os.path.join(base_dir, filename)
                self.__test_img(image_path, save_path)

    def __create_cityscape_submission(self, image_path, save_path):
        label_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        result = self.forward(image_path)
        width = self.configer.get('data', 'input_size')[0] // self.configer.get('network', 'stride')
        height = self.configer.get('data', 'input_size')[1] // self.configer.get('network', 'stride')
        label_dst = np.ones((height, width), dtype=np.uint8) * 255
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[result == i] = label_list[i]

        label_img = np.array(label_dst, dtype=np.uint8)
        label_img = Image.fromarray(label_img, 'P')
        label_img.save(save_path)

    def create_submission(self, test_dir=None):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/pose', self.configer.get('dataset'), 'submission')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for filename in self.__list_dir(test_dir):
            image_path = os.path.join(test_dir, filename)
            save_path = os.path.join(base_dir, filename)

            if self.configer.get('dataset') == 'cityscape':
                self.__create_cityscape_submission(image_path, save_path)

        else:
            Log.error('Dataset: {} is not valid.'.format(self.configer.get('dataset')))
            exit(1)

    def __list_dir(self, dir_name):
        filename_list = list()
        for item in os.listdir(dir_name):
            if os.path.isdir(item):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append('{}/{}'.format(item, filename))

            else:
                filename_list.append(item)

        return filename_list
