#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Main Scripts for computer vision.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from methods.method_selector import MethodSelector
from utils.configer import Configer
from utils.logger import Logger as Log


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The dir of model save path.')
    parser.add_argument('--save_name', default=None, type=str,
                        dest='save_name', help='The dir of model save path.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='base_lr', help='The learning rate.')
    parser.add_argument('--lr_policy', default='step', type=str,
                        dest='lr_policy', help='The policy of lr during training.')

    # ***********  Params for logging.  **********
    parser.add_argument('--log_level', default=None, type=str,
                        dest='log_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default=None, type=str,
                        dest='stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default=None, type=str,
                        dest='log_file', help='The path of log files.')

    # ***********  Params for test or submission.  **********
    parser.add_argument('--test_img', default=None, type=str,
                        dest='test_img', help='The test path of image.')
    parser.add_argument('--test_dir', default=None, type=str,
                        dest='test_dir', help='The test directory of images.')
    parser.add_argument('--test_set', default='val2017', type=str,
                        dest='test_set', help='The test set of images.')

    args = parser.parse_args()

    configer = Configer(args)

    Log.init(log_level=configer.get('logging', 'log_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    method_selector = MethodSelector(configer)
    model = None
    if configer.get('task') == 'pose':
        model = method_selector.select_pose_model()
    elif configer.get('task') == 'seg':
        model = method_selector.select_seg_model()
    elif configer.get('task') == 'det':
        model = method_selector.select_det_model()
    elif configer.get('task') == 'cls':
        model = method_selector.select_cls_model()
    else:
        Log.error('Method: {} is not valid.'.format(configer.get('method')))
        exit(1)

    model.init_model()

    if configer.get('phase') == 'train':
        model.train()
    elif configer.get('phase') == 'test' and configer.get('resume') is not None:
        model.test()
    elif configer.get('phase') == 'submission' and configer.get('resume') is not None:
        model.create_submission()
    else:
        Log.error('Phase: {} is not valid.'.format(args.phase))
        exit(1)
