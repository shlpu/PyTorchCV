#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Pose Estimation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, inputs, targets, masks=None, weights=None):
        loss = 0.0
        if isinstance(inputs, list) and weights is not None:
            for i in range(len(inputs)):
                if masks is not None:
                    loss += weights[i] * self.mse_loss(inputs[i]*masks, targets)
                else:
                    loss += weights[i] * self.mse_loss(inputs[i], targets)
        else:
            if masks is not None:
                loss = self.mse_loss(inputs*masks, targets)
            else:
                loss = self.mse_loss(inputs, targets)

        loss = loss / targets.size(0)
        return loss


class CapsuleLoss(nn.Module):

    def __init__(self, num_keypoints=18, l_vec=64):
        super(CapsuleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.num_keypoints = num_keypoints + 1
        self.l_vec = l_vec

    def forward(self, inputs, targets, masks=None, is_focal=False):
        inputs = inputs.view(inputs.size(0), self.num_keypoints,
                             self.l_vec, inputs.size(2), inputs.size(3))

        preds = torch.sqrt((inputs**2).sum(dim=2, keepdim=False))
        if masks is not None:
            preds = preds * masks

        if is_focal:
            loss = self.mse_loss(preds, targets)
        else:
            diff = preds - targets
            diff = diff ** 2
            alpha = 2.0
            weights = targets * alpha
            weights = torch.exp(weights)
            diff = weights * diff
            loss = diff.mean()

        return loss


class EmbeddingLoss(nn.Module):

    def __init__(self, num_keypoints=18,
                       l_vec=64):
        super(EmbeddingLoss, self).__init__()
        self.num_keypoints = num_keypoints + 1
        self.l_vec = l_vec
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, inputs, tags, numH):
        label = []
        vec1 = []
        vec2 = []
        batch_size = inputs.size(0)

        h_tag_means = [[Variable(torch.zeros(self.l_vec,), requires_grad=True).cuda()
                        for h in range(numH[b].numpy()[0])] for b in range(inputs.size()[0])]

        for b in range(batch_size):
            batch_heatmap = inputs[b]
            for n in range(numH[b].numpy()[0]):
                valik = 0
                for k in range(1, self.num_keypoints):
                    keypoint_heatmap = batch_heatmap[k*self.l_vec:(k+1)*self.l_vec]
                    tag = keypoint_heatmap.masked_select(tags[b][k].eq(n+1).unsqueeze(0))
                    # print tag.size()
                    if tag.size() != torch.Size([]):
                        h_tag_means[b][n] += tag
                        valik = valik + 1

                h_tag_means[b][n] = h_tag_means[b][n] / max(valik, 1)

        for b in range(batch_size):
            for n in range(numH[b].numpy()[0]):
                for k in range(1, self.num_keypoints):
                    tag = inputs[b][k*self.l_vec:(k+1)*self.l_vec]\
                            .masked_select(tags[b][k].eq(n+1).unsqueeze(0))
                    if tag.size() != torch.Size([]):
                        label.append(1)
                        vec1.append(tag)
                        vec2.append(h_tag_means[b][n])

        for b in range(batch_size):
            for n1 in range(numH[b].numpy()[0]):
                for n2 in range(numH[b].numpy()[0]):
                    if n1 != n2:
                        label.append(-1)
                        vec1.append(h_tag_means[b][n1])
                        vec2.append(h_tag_means[b][n2])

        pair_count = len(label)
        vector_1 = Variable(torch.zeros(pair_count, self.l_vec),
                            requires_grad=True).cuda()
        vector_2 = Variable(torch.zeros(pair_count, self.l_vec),
                            requires_grad=True).cuda()

        for i in range(pair_count):
            vector_1[i] = vec1[i]
            vector_2[i] = vec2[i]

        label = Variable(torch.FloatTensor(label), requires_grad=True).cuda()
        if pair_count == 0:
            loss = 0.0
        else:
            loss = self.cosine_loss(vector_1, vector_2, label)

        return loss


class MarginLoss(nn.Module):

    def __init__(self, num_keypoints=18, l_vec=64):
        super(MarginLoss, self).__init__()
        self.num_keypoints = num_keypoints + 1
        self.l_vec = l_vec

    def forward(self, inputs, targets, mask, size_average=True):
        batch_size = inputs.size(0)
        inputs = inputs.view(inputs.size(0), self.num_keypoints,
                             self.l_vec, inputs.size(2), inputs.size(3))
        # ||vc|| from the paper.
        v_mag = torch.sqrt((inputs**2).sum(dim=2, keepdim=False))
        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero)**2
        # max_r.max() may be much bigger.
        max_r = torch.max(v_mag - m_minus, zero)**2
        # This is equation 4 from the paper.
        loss_lambda = 1.0
        T_c = targets
        # bugs when targets!=0 or 1
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        if size_average:
            L_c = L_c.mean()

        return L_c


class VoteLoss(nn.Module):
    def __init__(self):
        super(VoteLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        inputs = torch.sqrt((inputs ** 2).sum(dim=-1, keepdim=False))
        loss = self.mse_loss(inputs, targets)
        return loss