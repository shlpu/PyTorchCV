#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from datasets.seg.cityscape.fs_cityscape_loader import FSCityScapeLoader
from datasets.seg_data_loader import SegDataLoader
from loss.seg_loss_manager import SegLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from models.seg_model_manager import SegModelManager
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
from vis.visualizer.seg_visualizer import SegVisualizer


class FCNSegmentor(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_loss_manager = SegLossManager(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.seg_model_manager = SegModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.lr = None
        self.iters = None

    def init_model(self):
        self.seg_net = self.seg_model_manager.seg_net()
        self.iters = 0
        self.seg_net, _ = self.module_utilizer.load_net(self.seg_net)

        self.optimizer, self.lr = self.module_utilizer.update_optimizer(self.seg_net, self.iters)

        if self.configer.get('dataset') == 'cityscape':
            self.train_loader = self.seg_data_loader.get_trainloader(FSCityScapeLoader)
            self.val_loader = self.seg_data_loader.get_valloader(FSCityScapeLoader)

        else:
            Log.error('Dataset: {} is not valid!'.format(self.configer.get('dataset')))
            exit(1)

        self.pixel_loss = self.seg_loss_manager.get_seg_loss('cross_entropy_loss')

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        start_time = time.time()

        # data_tuple: (inputs, heatmap, maskmap, tagmap, num_objects)
        for i, data_tuple in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            if len(data_tuple) < 2:
                Log.error('Train Loader Error!')
                exit(0)

            inputs = Variable(data_tuple[0].cuda(async=True))
            heatmap = Variable(data_tuple[1].cuda(async=True))
            maskmap = None
            if len(data_tuple) > 2:
                maskmap = Variable(data_tuple[2].cuda(async=True))

            # Forward pass.
            outputs = self.seg_net(inputs)

            # Compute the loss of the train batch & backward.
            loss_pixel = self.pixel_loss(outputs, heatmap, maskmap)
            loss = loss_pixel

            self.train_losses.update(loss.data[0], inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.iters += 1

            # Print the log info & reset the states.
            if self.iters % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Iteration: {0}\t'
                         'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {2}\n'
                         'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                         self.iters, self.configer.get('solver', 'display_iter'),
                         self.lr, batch_time=self.batch_time,
                         data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # Check to val the current model.
            if self.val_loader is not None and \
               self.iters % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

            self.optimizer, self.lr = self.module_utilizer.update_optimizer(self.seg_net, self.iters)

    def __val(self):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()

        for j, data_tuple in enumerate(self.val_loader):
            # Change the data type.
            inputs = Variable(data_tuple[0].cuda(async=True), volatile=True)
            targets = Variable(data_tuple[1].cuda(async=True), volatile=True)
            maskmap = None
            if len(data_tuple) > 2:
                maskmap = Variable(data_tuple[2].cuda(async=True), volatile=True)

            # Forward pass.
            outputs = self.seg_net(inputs)

            # Compute the loss of the val batch.
            loss_pixel = self.pixel_loss(outputs, targets, maskmap)
            loss = loss_pixel

            self.val_losses.update(loss.data[0], inputs.size(0))

            # Update the vars of the val phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.module_utilizer.save_net(self.seg_net, self.iters)
        # Print the log info & reset the states.
        Log.info(
            'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
            'Loss {loss.avg:.8f}\n'.format(
            batch_time=self.batch_time, loss=self.val_losses))
        self.batch_time.reset()
        self.val_losses.reset()
        self.seg_net.train()

    def train(self):
        cudnn.benchmark = True
        while self.iters < self.configer.get('solver', 'max_iter'):
            self.__train()
            if self.iters == self.configer.get('solver', 'max_iter'):
                break


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
