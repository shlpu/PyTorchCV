#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Configer class for all hyper parameters.


import json
import os
import argparse


class Configer(object):

    def __init__(self, args):
        self.args_dict = args.__dict__
        self.params_root = None

        if not os.path.exists(args.hypes):
            print('Json Path:{} not exists!'.format(json_path))
            exit(0)

        json_stream = open(args.hypes, 'r')
        self.params_root = json.load(json_stream)
        json_stream.close()

    def get(self, *key):
        if len(key) == 0:
            return self.params_root

        elif len(key) == 1:
            if key[0] in self.args_dict and self.args_dict[key[0]] is not None:
                return self.args_dict[key[0]]
            elif key[0] in self.args_dict and key[0] not in self.params_root:
                return None
            else:
                return self.params_root[key[0]]

        elif len(key) == 2:
            if key[1] in self.args_dict and self.args_dict[key[1]] is not None:
                return self.args_dict[key[1]]
            elif key[1] in self.args_dict and key[1] not in self.params_root[key[0]]:
                return None
            else:
                return self.params_root[key[0]][key[1]]

        else:
            print('KeyError: {}.'.format(key))
            exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default='../hypes/road_pose.json', type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of Pose Estimator.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')

    args = parser.parse_args()

    configer = Configer(args)
    print configer.get()
    print configer.get('resume')
    print configer.get('logging', 'save_iter')