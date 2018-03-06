#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualize the tensor of the pose estimator.


import cv2
import numpy as np
import os
import math
import matplotlib
import pylab as plt
from numpy import ma
from scipy.ndimage.filters import gaussian_filter

from datasets.tools.transforms import DeNormalize
from utils.logger import Logger as Log


PEAK_DIR = 'vis/results/pose/peaks'
PAF_DIR = 'vis/results/pose/pafs'
TENSOR_DIR = 'vis/results/pose/tensors'
POSE_DIR = 'vis/results/pose/poses'
LIMB_DIR = 'vis/results/pose/limbs'


class PoseVisualizer(object):

    def __init__(self, configer):
        self.configer = configer

    def __get_peaks(self, heatmap):
        # print (self.configer.get('input_size')[0], self.configer.get('input_size'))
        heatmap = cv2.resize(heatmap, (self.configer.get('data', 'input_size')[0],
                                       self.configer.get('data', 'input_size')[1]), interpolation=cv2.INTER_CUBIC)
        s_map = gaussian_filter(heatmap, sigma=3)
        map_left = np.zeros(s_map.shape)
        map_left[:, 1:] = s_map[:, :-1]
        map_right = np.zeros(s_map.shape)
        map_right[:, :-1] = s_map[:, 1:]
        map_up = np.zeros(s_map.shape)
        map_up[1:, :] = s_map[:-1, :]
        map_down = np.zeros(s_map.shape)
        map_down[:-1, :] = s_map[1:, :]

        # Get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (s_map >= map_left, s_map >= map_right,
             s_map >= map_up, s_map >= map_down,
             s_map > self.configer.get('vis', 'part_threshold')))

        peaks = list(zip(np.nonzero(peaks_binary)[1],
                         np.nonzero(peaks_binary)[0]))

        # A point format: (w, h, score, number)
        peaks_with_score = [x + (s_map[x[1], x[0]],) for x in peaks]
        return peaks_with_score

    def vis_peaks(self, heatmap, ori_img, name='default',
                  vis_dir=PEAK_DIR, scale_factor=1, img_size=(368, 368)):
        vis_dir = os.path.join(self.configer.get('project_dir'), vis_dir)
        if not os.path.exists(vis_dir):
            Log.error('Dir:{} not exists!'.format(vis_dir))
            os.makedirs(vis_dir)

        if not isinstance(heatmap, np.ndarray):
            if len(heatmap.size()) != 3:
                Log.error('Heatmap size is not valid.')
                exit(1)

            heatmap = heatmap.data.squeeze().cpu().numpy().transpose(1, 2, 0)

        if not isinstance(ori_img, np.ndarray):
            ori_img = DeNormalize(mean=[128.0, 128.0, 128.0],std=[256.0, 256.0, 256.0])(ori_img)
            ori_img = ori_img.data.cpu().squeeze().numpy().transpose(1, 2, 0)

        for j in range(self.configer.get('num_keypoints')):
            peaks = self.__get_peaks(heatmap[:, :, j].data.cpu().numpy())
            image_path = os.path.join(vis_dir, '{}_{}.jpg'.format(name, j))
            for peak in peaks:
                image = cv2.circle(ori_img, (peak[0], peak[1]),
                                   self.configer.get('vis', 'circle_radius'), (0,255,0), thickness=-1)
                image = self.scale_image(image, scale_factor, img_size)
                cv2.imwrite(image_path, image)

    def vis_paf(self, inputs, ori_img, name='default', vis_dir=PAF_DIR):
        vis_dir = os.path.join(self.configer.get('project_dir'), vis_dir)
        if not os.path.exists(vis_dir):
            Log.error('Dir:{} not exists!'.format(vis_dir))
            os.makedirs(vis_dir)

        if not isinstance(inputs, np.ndarray):
            if len(inputs.size()) != 3:
                Log.error('Heatmap size is not valid.')
                exit(1)

            inputs = inputs.data.squeeze().cpu().numpy().transpose(1, 2, 0)

        if not isinstance(ori_img, np.ndarray):
            if len(ori_img.size()) != 3:
                Log.error('Heatmap size is not valid.')
                exit(1)

            ori_img = DeNormalize(mean=[128.0, 128.0, 128.0],std=[256.0, 256.0, 256.0])(ori_img)
            ori_img = ori_img.data.cpu().squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)

        if inputs.shape[0] != ori_img.shape[0]:
            resize_width = (inputs.shape[1] + ori_img.shape[1]) // 2
            resize_height = (inputs.shape[0] + ori_img.shape[0]) // 2
            inputs = cv2.resize(inputs, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
            ori_img = cv2.resize(ori_img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

        for i in range(1):
            U = inputs[:, :, 2*i] * -1
            V = inputs[:, :, 2*i+1]
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')
            M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

            # 1
            plt.figure()
            plt.imshow(ori_img+0.5, alpha=.5)
            s = 5
            Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
                           scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(10, 10)
            plt.savefig(os.path.join(vis_dir, '{}_{}.jpg'.format(name, i)))

    def vis_tensor(self, tensor_in, name='default',
                   vis_dir=TENSOR_DIR, scale_factor=1, img_size=(368, 368)):
        vis_dir = os.path.join(self.configer.get('project_dir'), vis_dir)

        tensor = tensor_in.clone()
        if not os.path.exists(vis_dir):
            Log.error('Dir:{} not exists!'.format(vis_dir))
            os.makedirs(vis_dir)

        if len(tensor.size()) == 4:
            for i in range(tensor.size(0)):
                heatmap = tensor[i,:,:,:].data.cpu().numpy()
                heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
                for j in range(tensor.size(1)):
                    image = heatmap[:,:,j:j+1]*255
                    image = self.scale_image(image, scale_factor, img_size)
                    img_path = os.path.join(vis_dir, '{}_{}_{}.jpg'.format(name, i, j))
                    cv2.imwrite(img_path, image)

    def vis_poses(self, image, kpts_list,
                  name='default', vis_dir=POSE_DIR,
                  scale_factor=1, img_size=None,  part='all'):
        """
          Show the diff parts of persons.
        """
        vis_dir = os.path.join(self.configer.get('project_dir'), vis_dir)

        if not os.path.exists(vis_dir):
            Log.error('Dir:{} not exists!'.format(vis_dir))
            os.makedirs(vis_dir)

        img_path = os.path.join(vis_dir, '{}_{}.jpg'.format(name, part))

        for kpts in kpts_list:
            if part == 'all':
                for i in range(self.configer.get('num_keypoints')):
                    if -1 in kpts[i]:
                        continue

                    image = cv2.circle(image, kpts[i], self.configer.get('vis', 'circle_radius'),
                                       self.configer.get('coco', 'color_list')[i], thickness=-1)
            else:
                if not self.configer.get('coco', 'pose_id_dict').has_key(part):
                    Log.error('Part {} is not valid!'.format(part))
                    continue

                part_index = self.configer.get('coco', 'pose_id_dict')[part] -1
                if kpts[part_index] == -1:
                    continue

                image = cv2.circle(image, kpts[i], self.configer.get('vis', 'circle_radius'),
                                   self.configer.get('coco', 'color_list')[i], thickness=-1)

        image = self.scale_image(image, scale_factor, img_size)
        cv2.imwrite(img_path, image)

    def vis_limbs(self, image, kpts_list,
                  name='default', vis_dir=LIMB_DIR,
                  scale_factor=1, img_size=None, limb="all"):
        """
          Show the diff limbs of persons.
          Args:
            Limb format: nose2neck.
        """
        vis_dir = os.path.join(self.configer.get('project_dir'), vis_dir)

        if not os.path.exists(vis_dir):
            Log.error('Dir:{} not exists!'.format(vis_dir))
            os.makedirs(vis_dir)

        img_path = os.path.join(vis_dir, '{}_{}.jpg'.format(name, limb))

        for kpts in kpts_list:
            if limb == 'all':
                for i in range(self.configer.get('coco', 'limb_seq')):
                    from_node = self.configer.get('coco', 'limb_seq')[i][0] - 1
                    to_node = self.configer.get('coco', 'limb_seq')[i][1] - 1
                    if -1 in kpts[from_node] or -1 in kpts[to_node]:
                        continue

                    mx = (kpts[from_node][0] + kpts[to_node][0]) / 2.0
                    my = (kpts[from_node][1] + kpts[to_node][1]) / 2.0
                    length = ((kpts[from_node][0] - kpts[to_node][0]) ** 2
                              + (kpts[from_node][1] - kpts[to_node][1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(kpts[from_node][0] - kpts[to_node][0],
                                                    kpts[from_node][1] - kpts[to_node][1]))
                    center = (int(mx), int(my))

                    polygon = cv2.ellipse2Poly(center, int(length / 2.0), self.configer.get('vis', 'stick_width'),
                                               angle, 0, 360, 1)
                    image = cv2.fillConvexPoly(image, polygon, self.configer.get('coco', 'color_list')[i])
            else:
                two_parts = limb.split('2')
                from_part = two_parts[0]
                to_part = two_parts[1]

                if not self.configer.get('coco', 'pose_id_dict').has_key(from_part) or \
                   not self.configer.get('coco', 'pose_id_dict').has_key(to_part):
                    Log.error('Limb {} is not valid!'.format(limb))
                    continue

                from_node = self.configer.get('coco', 'pose_id_dict')[from_part] - 1
                to_node = self.configer.get('coco', 'pose_id_dict')[to_part] - 1
                if -1 in kpts[from_node] or -1 in kpts[to_node]:
                    continue

                mx = (kpts[from_node][0] + kpts[to_node][0]) / 2.0
                my = (kpts[from_node][1] + kpts[to_node][1]) / 2.0
                length = ((kpts[from_node][0] - kpts[to_node][0]) ** 2
                          + (kpts[from_node][1] - kpts[to_node][1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(kpts[from_node][0] - kpts[to_node][0],
                                                kpts[from_node][1] - kpts[to_node][1]))
                center = (int(mx), int(my))

                polygon = cv2.ellipse2Poly(center, int(length / 2.0),
                                           self.configer.get('vis', 'stick_width'), angle, 0, 360, 1)
                image = cv2.fillConvexPoly(image, polygon, self.configer.get('coco', 'color_list')[i])

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


if __name__ == "__main__":
    # Test the visualizer.
    pass