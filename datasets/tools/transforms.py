#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import numpy as np
import numbers
import collections
import cv2
from PIL import Image
import matplotlib

from utils.logger import Logger as Log


class Normalize(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs


class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        result = inputs.clone()
        for i in range(result.size(1)):
            result[:,i,:,:] = result[:,i,:,:] * self.std[i] + self.mean[i]

        return result


class ToTensor(object):
    """Convert a ``numpy.ndarray or Image`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        inputs (numpy.ndarray or Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            inputs = torch.from_numpy(np.array(inputs).transpose(2, 0, 1))
        else:
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))

        return inputs.float()


class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs


class PadImage(object):
    """ Padding the Image to proper size.
        Args:
            stride: the stride of the network.
            pad_value: the value that pad to the image border.
            img: np.array object as input.

        Returns:
            img: np.array object.
    """
    def __init__(self, stride, pad_value):
        self.stride = stride
        self.pad_value = pad_value

    def __call__(self, img):
        img = np.array(img)
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % self.stride == 0) else self.stride - (h % self.stride)  # down
        pad[3] = 0 if (w % self.stride == 0) else self.stride - (w % self.stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + self.pad_value, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + self.pad_value, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + self.pad_value, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + self.pad_value, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)
        return img_padded, pad


class RandomFlip(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, img, label=None, mask=None, kpt=None, center=None):
        rand_value = random.randint(0, 2 // self.ratio)
        if rand_value == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if label is not None:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if rand_value == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if label is not None:
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, label, mask, kpt, center


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
    """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, img, label=None, mask=None, kpt=None, center=None):
        img = np.array(img)
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
        return Image.fromarray(img_new), label, mask, kpt, center


class RandomResize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_min=0.5, scale_max=1.1, size=None):
        self.scale_min = scale_min
        self.scale_max = scale_max

        if size is not None:
            if isinstance(size, int):
                self.size = (size, size)
            elif isinstance(size, collections.Iterable) and len(size) == 2:
                self.size = size
            else:
                raise TypeError('Got inappropriate size arg: {}'.format(size))
        else:
            self.size = None

    def __call__(self, img, label=None, mask=None, kpt=None, center=None):
        """
        Args:
            img     (Image):   Image to be resized.
            mask    (Image):   Mask to be resized.
            kpt     (list):    keypoints to be resized.
            center: (list):    center points to be resized.

        Returns:
            Image:  Randomly resize image.
            Image:  Randomly resize mask.
            list:   Randomly resize keypoints.
            list:   Randomly resize center points.
        """
        width, height = img.size
        w_scale_ratio = random.uniform(self.scale_min, self.scale_max)
        h_scale_ratio = random.uniform(self.scale_min, self.scale_max)
        if self.size is not None:
            w_scale_ratio = self.size[0] / width
            h_scale_ratio = self.size[1] / height

        if kpt is not None and kpt != list():
            num_objects = len(kpt)
            num_keypoints = len(kpt[0])

            for i in range(num_objects):
                for j in range(num_keypoints):
                    kpt[i][j][0] *= w_scale_ratio
                    kpt[i][j][1] *= h_scale_ratio

                if center is not None:
                    center[i][0] *= w_scale_ratio
                    center[i][1] *= h_scale_ratio

        if self.size is not None:
            converted_size = self.size
        else:
            converted_size = (int(width*w_scale_ratio), int(height*h_scale_ratio))

        img = img.resize(converted_size, Image.BILINEAR)
        if label is not None:
            label = label.resize(converted_size, Image.NEAREST)
        if mask is not None:
            mask = mask.resize(converted_size, Image.NEAREST)

        return img, label, mask, kpt, center


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree

    def __call__(self, image, label=None, mask=None, kpt=None, center=None):
        """
        Args:
            img    (Image):     Image to be rotated.
            mask   (Image):     Mask to be rotated.
            kpt    (list):      Keypoints to be rotated.
            center (list):      Center points to be rotated.

        Returns:
            Image:     Rotated image.
            list:      Rotated key points.
        """
        rotate_degree = random.uniform(-self.max_degree, self.max_degree)

        image = np.array(image)
        height, width, _ = image.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        Log.info('{}_{}'.format(new_width, new_height))
        image = cv2.warpAffine(image, rotate_mat, (new_width, new_height), borderValue=(128, 128, 128))
        image = Image.fromarray(image)
        if label is not None:
            label = np.array(label)
            label = cv2.warpAffine(label, rotate_mat, (new_width, new_height), borderValue=(255, 255, 255))
            label = Image.fromarray(label)

        if mask is not None:
            mask = np.array(mask)
            mask = cv2.warpAffine(mask, rotate_mat, (new_width, new_height), borderValue=(1, 1, 1))
            mask = Image.fromarray(mask)

        if kpt is not None:
            num_objects = len(kpt)
            num_keypoints = len(kpt[0])
            for i in range(num_objects):
                for j in range(num_keypoints):
                    x = kpt[i][j][0]
                    y = kpt[i][j][1]
                    p = np.array([x, y, 1])
                    p = rotate_mat.dot(p)
                    kpt[i][j][0] = p[0]
                    kpt[i][j][1] = p[1]

                if center is not None:
                    x = center[i][0]
                    y = center[i][1]
                    p = np.array([x, y, 1])
                    p = rotate_mat.dot(p)
                    center[i][0] = p[0]
                    center[i][1] = p[1]

        return image, label, mask, kpt, center


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, collections.Iterable) and len(size) == 2:
            self.size = size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(size))

    def __call__(self, img, label=None, mask=None, kpt=None, center=None):
        """
        Args:
            img (Image):   Image to be cropped.
            mask (Image):  Mask to be cropped.
            kpt (list):    keypoints to be cropped.
            center (list): center points to be cropped.

        Returns:
            Image:  Cropped image.
            Image:  Cropped mask.
            list:   Cropped keypoints.
            list:   Cropped center points.
        """
        width, height = img.size
        if self.size[0] > width or self.size[1] > height:
            return img, label, mask, kpt, center

        offset_left = random.randint(0, width-self.size[0])
        offset_up = random.randint(0, height - self.size[1])

        if kpt is not None:
            num_objects = len(kpt)
            num_keypoints = len(kpt[0])

            for i in range(num_objects):
                for j in range(num_keypoints):
                    kpt[i][j][0] -= offset_left
                    kpt[i][j][1] -= offset_up

                if center is not None:
                    center[i][0] -= offset_left
                    center[i][1] -= offset_up

        img = img.crop((offset_left, offset_up, offset_left+self.size[0], offset_up+self.size[1]))
        if label is not None:
            label = label.crop((offset_left, offset_up, offset_left+self.size[0], offset_up+self.size[1]))

        if mask is not None:
            mask = mask.crop((offset_left, offset_up, offset_left+self.size[0], offset_up+self.size[1]))
        return img, label, mask, kpt, center


class BaseCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> BaseCompose([
        >>>     RandomCrop(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=None, mask=None, kpt=None, center=None):

        for t in self.transforms:
            img, label, mask, kpt, center = t(img, label, mask, kpt, center)

        if label is None and mask is None and kpt is None and center is None:
            return img
        elif mask is None and kpt is None and center is None:
            return img, label
        elif label is None:
            return img, mask, kpt, center
        else:
            return img, label, mask, kpt, center


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs
