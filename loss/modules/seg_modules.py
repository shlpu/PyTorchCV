#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets, maskmap=None):
        return self.nll_loss(F.log_softmax(inputs), targets)


# Cosine Embedding loss.
class EmbeddingLoss(nn.Module):
    def __init__(self, num_classes):
        super(EmbeddingLoss, self).__init__()
        self.num_classes = num_classes
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets):
        embedding_loss = 0.0
        inputs = inputs.transpose(0, 1)
        center_list = []
        for i in range(self.num_classes):
            mask = self.get_mask(targets, i)
            sum_pixel = max(mask.sum(), 1)
            # print sum_pixel
            mask_ = Variable(torch.cuda.FloatTensor(inputs.size()))
            for j in range(inputs.size()[0]):
                mask_[j] = mask
            center = inputs * mask_
            center = torch.sum(center.view(center.size()[0], -1), 1)
            center = center / sum_pixel
            center_list.append(center)

        center_array = Variable(torch.zeros(self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
        item_count = 0
        for center in center_list:
            center_array[item_count] = center
            item_count = item_count + 1
        for i in range(self.num_classes):
            label = Variable(torch.zeros(self.num_classes, ).type(torch.IntTensor)).cuda()
            center_dual = Variable(torch.zeros(self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
            for k in range(self.num_classes):
                center_dual[k] = center_list[i]

            for j in range(self.num_classes):
                if j == i:
                    label[j] = 1
                else:
                    label[j] = -1
            embedding_loss += self.cosine_loss(center_array, center_dual, label)

        # print embedding_loss.requires_grad
        return embedding_loss / (self.num_classes * self.num_classes)

    def get_mask(self, targets, i):
        targets_cp = torch.cuda.FloatTensor(targets.size())
        targets_cp.copy_(targets.data)
        if i == 0:
            targets_cp[targets_cp != 0] = 2
            targets_cp[targets_cp == 0] = 1
            targets_cp[targets_cp == 2] = 0
        else:
            targets_cp[targets_cp != i] = 0
            targets_cp[targets_cp == i] = 1

        return targets_cp


class IOULoss(nn.Module):
    def __init__(self, num_classes):
        super(IOULoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        iou_loss = 0  # Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).cuda()
        predicts = Variable(inputs.data.max(1)[1])
        inputs = F.softmax(inputs)

        for i in range(self.num_classes):
            union_mask, intersect_mask = self.get_class_loss(predicts, targets, i)
            union = inputs[:, i:i + 1] * Variable(union_mask)
            intersect = inputs[:, i:i + 1] * Variable(intersect_mask)
            union = torch.sum(union.view(union.size()[0], -1), 1)
            intersect = torch.sum(intersect.view(intersect.size()[0], -1), 1)
            label = Variable(torch.ones(union.size()[0], ).type(torch.FloatTensor)).cuda()
            loss_map = intersect / torch.max(union, label)
            class_iou_loss = self.mse_loss(loss_map, label)
            iou_loss += class_iou_loss
            # print "Class %d: %.4f" % (i, class_iou_loss.data[0])

        return iou_loss

    def get_class_loss(self, predicts, targets, i):
        predicts_cp = torch.cuda.FloatTensor(predicts.size())
        targets_cp = torch.cuda.FloatTensor(targets.size())
        predicts_cp.copy_(predicts.data)
        targets_cp.copy_(targets.data)
        if i == 0:
            predicts_cp[predicts_cp != 0] = 2
            predicts_cp[predicts_cp == 0] = 1
            predicts_cp[predicts_cp == 2] = 0
            targets_cp[targets_cp != 0] = 2
            targets_cp[targets_cp == 0] = 1
            targets_cp[targets_cp == 2] = 0
        else:
            predicts_cp[predicts_cp != i] = 0
            predicts_cp[predicts_cp == i] = 1
            targets_cp[targets_cp != i] = 0
            targets_cp[targets_cp == i] = 1

        union = predicts_cp + targets_cp
        intersect = predicts_cp * targets_cp
        union[union == 2] = 1
        return union, intersect


class FocalLoss(nn.Module):
    def __init__(self, y):
        super(FocalLoss, self).__init__()
        self.y = y

    def forward(self, output, target):
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1, torch.unsqueeze(target, 1))
        focus_p = torch.pow(1 - Pt, self.y)
        alpha = 0.25
        nll_feature = -f_out.gather(1, torch.unsqueeze(target, 1))
        weight_nll = alpha * focus_p * nll_feature
        loss = weight_nll.mean()
        return loss
