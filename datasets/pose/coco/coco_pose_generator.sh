#!/usr/bin bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/deepmotion/PycharmProjects/PoseEstimation'

INPUT_SIZE=368

COCO_TRAIN_IMG_DIR='/home/deepmotion/DataSet/MSCOCO/train2017'
COCO_VAL_IMG_DIR='/home/deepmotion/DataSet/MSCOCO/val2017'

COCO_ANNO_DIR='/home/deepmotion/DataSet/MSCOCO/annotations/'
TRAIN_ANNO_FILE=${COCO_ANNO_DIR}'person_keypoints_train2017.json'
VAL_ANNO_FILE=${COCO_ANNO_DIR}'person_keypoints_val2017.json'

TRAIN_ROOT_DIR='/home/deepmotion/DataSet/COCO/train'
VAL_ROOT_DIR='/home/deepmotion/DataSet/COCO/val'


python2.7 coco_pose_generator.py --root_dir $TRAIN_ROOT_DIR \
                                 --anno_file $TRAIN_ANNO_FILE \
                                 --img_dir $COCO_TRAIN_IMG_DIR \
                                 --input_size $INPUT_SIZE

python2.7 coco_pose_generator.py --root_dir $VAL_ROOT_DIR \
                                 --anno_file $VAL_ANNO_FILE \
                                 --img_dir $COCO_VAL_IMG_DIR \
                                 --input_size $INPUT_SIZE
