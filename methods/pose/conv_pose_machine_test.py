#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Test class for convolutional pose machine.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import torch
from torch.autograd import Variable

from datasets.tools.transforms import PadImage
from models.pose_model_manager import PoseModelManager
from methods.tools.module_utilizer import ModuleUtilizer
from vis.visualizer.pose_visualizer import PoseVisualizer
from utils.logger import Logger as Log


class ConvPoseMachineTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.pose_vis = PoseVisualizer(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.pose_net = None

    def init_model(self):
        self.pose_net = self.pose_model_manager.pose_detector()
        self.pose_net, _ = self.module_utilizer.load_net(self.pose_net)
        self.pose_net.eval()

    def __test_img(self, image_path, save_path):
        image_raw = cv2.imread(image_path)
        heatmap_avg = self.__get_heatmap(image_raw)
        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        image_save = self.__draw_key_point(all_peaks, image_raw)
        cv2.imwrite(save_path, image_save)

    def __get_heatmap(self, img_raw):
        multiplier = [scale * self.configer.get('data', 'input_size')[0] / img_raw.shape[1]
                      for scale in self.configer.get('data', 'scale_search')]

        heatmap_avg = np.zeros(img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'heatmap_out'))

        for i, scale in enumerate(multiplier):
            img_test = cv2.resize(img_raw, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_test_pad, pad = PadImage(self.configer.get('network', 'stride'), 0)(img_test)
            img_test_pad = np.transpose(np.float32(img_test_pad[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5

            feed = Variable(torch.from_numpy(img_test_pad)).cuda()
            heatmap = self.pose_net(feed)
            # extract outputs, resize, and remove padding
            heatmap = heatmap.data.squeeze().cpu().transpose(1, 2, 0)
            heatmap = cv2.resize(heatmap, (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)

            heatmap = heatmap[:img_test_pad.shape[0] - pad[2], :img_test_pad.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        return heatmap_avg

    def __extract_heatmap_info(self, heatmap_avg):
        all_peaks = []

        for part in range(self.configer.get('data', 'num_keypoints')):
            map_ori = heatmap_avg[:, :, part]
            map_gau = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map_gau.shape)
            map_left[1:, :] = map_gau[:-1, :]
            map_right = np.zeros(map_gau.shape)
            map_right[:-1, :] = map_gau[1:, :]
            map_up = np.zeros(map_gau.shape)
            map_up[:, 1:] = map_gau[:, :-1]
            map_down = np.zeros(map_gau.shape)
            map_down[:, :-1] = map_gau[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
                 map_gau >= map_down, map_gau > self.configer.get('vis', 'part_threshold')))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            all_peaks.append(peaks_with_score)

        return all_peaks

    def __draw_key_point(self, all_peaks, img_raw):
        img_canvas = img_raw.copy()  # B,G,R order

        for i in range(self.configer.get('data', 'num_keypoints')):
            for j in range(len(all_peaks[i])):
                cv2.circle(img_canvas, all_peaks[i][j][0:2], self.configer.get('vis', 'stick_width'),
                           self.configer.get('details', 'color_list')[i], thickness=-1)

        return img_canvas

    def test(self, test_img=None, test_dir=None):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/pose', self.configer.get('dataset'), 'test')
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

    def __create_coco_submission(self, test_dir=None, base_dir=None):
        pass

    def create_submission(self, test_dir=None):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/pose', self.configer.get('dataset'), 'submission')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if self.configer.get('dataset') == 'coco':
            self.__create_coco_submission(test_dir)
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