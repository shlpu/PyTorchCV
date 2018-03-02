#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


import os
import torch
import torch.nn as nn

from utils.logger import Logger as Log


class ModuleUtilizer(object):

    def __init__(self, configer):
        self.configer = configer

    def update_optimizer(self, net, iters):
        """
          Adjust the learning rate during the train phase.
        """
        policy = self.configer.get('solver', 'lr_policy')
        lr = self.configer.get('solver', 'base_lr')
        if policy == 'fixed':
            lr = self.configer.get('solver', 'base_lr')

        elif policy == 'step':
            gamma = self.configer.get('solver', 'gamma')
            ratio = gamma ** (iters // self.configer.get('solver', 'step_size'))
            lr = self.configer.get('solver', 'base_lr') * ratio

        elif policy == 'exp':
            lr = self.configer.get('solver', 'base_lr') * (self.configer.get('solver', 'gamma') ** iters)

        elif policy == 'inv':
            power = -self.configer.get('solver', 'power')
            ratio = (1 + self.configer.get('solver', 'gamma') * iters) ** power
            lr = self.configer.get('solver', 'base_lr') * ratio

        elif policy == 'multistep':
            lr = self.configer.get('solver', 'base_lr')
            for step_value in self.configer.get('solver', 'stepvalue'):
                if iters >= step_value:
                    lr *= self.configer.get('solver', 'gamma')
                else:
                    break

        else:
            Log.error('Policy:{} is not valid.'.format(policy))
            exit(1)

        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.configer.get('solver', 'momentum'),
                                    weight_decay=self.configer.get('solver', 'weight_decay'))
        return optimizer, lr

    def load_net(self, net):
        net = nn.DataParallel(net, device_ids=self.configer.get('gpu')).cuda()
        iters = 0
        if self.configer.get('resume') is not None:
            checkpoint_dict = torch.load(self.configer.get('resume'))
            load_dict = dict()
            for key, value in checkpoint_dict['state_dict'].items():
                if key.split('.')[0] == 'module':
                    load_dict[key] = checkpoint_dict['state_dict'][key]
                else:
                    load_dict['module.{}'.format(key)] = checkpoint_dict['state_dict'][key]

            net.load_state_dict(load_dict)
            iters = checkpoint_dict['iter']

        return net, iters

    def save_net(self, net, iters):
        if iters % self.configer.get('checkpoints', 'save_iters') != 0:
            return

        state = {
            'iter': iters,
            'state_dict': net.state_dict(),
        }
        checkpoints_dir = self.configer.get('checkpoints', 'save_dir')

        latest_name = '{}_{}.pth'.format(self.configer.get('checkpoints', 'save_name'), iters)
        torch.save(state, os.path.join(checkpoints_dir, latest_name))
