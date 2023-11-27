#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=9):
        super(PointNet, self).__init__()
        self.args = args
        self.k = output_channels
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        #self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        #self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)

        self.conv6 = nn.Conv1d(1088, 512, kernel_size=1, bias=False)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv9 = nn.Conv1d(128, output_channels, kernel_size=1, bias=False)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn5 = nn.BatchNorm1d(256)


        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        # self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout()
        # self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        
        org_x = x.clone()
        batch = x.size()[0]
        npts = x.size()[2]
        #print("input 1 output shape = ", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print("input 2 output shape = ", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print("input 3 output shape = ", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print("input 4 output shape = ", x.shape)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1,1024)

        x = F.relu(self.bn4(self.fc1(x)))
        #print("input 5 output shape = ", x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        #print("input 6 output shape = ", x.shape)
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batch, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        #print("input 6a output shape = ", x.shape)

        org_x = org_x.transpose(2,1)
        #print("input 7 output shape = ", org_x.shape)
        x = torch.bmm(org_x, x)
        #print("input 8 output shape = ", x.shape)
        x = x.transpose(2,1)
        #print("input 9 output shape = ", x.shape)


        #print("input 1 output shape = ", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print("input 2 output shape = ", x.shape)
        org_x = x
        x = F.relu(self.bn2(self.conv2(x)))
        #print("input 3 output shape = ", x.shape)
        x = self.bn3(self.conv3(x))
        #print("input 4 output shape = ", x.shape)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1,1024)

        x = x.view(-1, 1024, 1).repeat(1,1,npts)
        x = torch.cat([x,org_x], 1)


        # Higher level layers
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        x = self.conv9(x)

        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batch, npts, self.k)
        #print("input 10 output shape = ", x.shape)
        #x = F.adaptive_max_pool2d(x, 1024).squeeze()
        #x = torch.max(x, 2, keepdim=True)[0]
        #x = x.view(-1, 1024)
        #print("input 7 output shape = ", x.shape)
        #x = self.linear1(x)
        #print("input 8 output shape = ", x.shape)
        #x = self.bn6(x)
        #print("input 9 output shape = ", x.shape)
        #x = F.relu(x)
        #print("input 10 output shape = ", x.shape)
        #x = F.relu(self.bn6(self.linear1(x)))
        #print("input 18 output shape = ", x.shape)
        #x = self.dp1(x)
        #print("input 19 output shape = ", x.shape)
        #x = self.linear2(x)
        #print("input 10 output shape = ", x.shape)
        #x = x.transpose(2, 1).contiguous()
        #print("transpose shape = ",x.shape)
        #x = x.transpose(-2, -1).contiguous()
        #print("transpose shape = ",x.shape)
        return x


class DGCNN(nn.Module):
    def __init__(self,args, num_classes=9):
        super(DGCNN, self).__init__()

        self.k = args.k
        self.emb_dims=args.emb_dims
        self.dropout=args.dropout

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        bs = x.size(0)
        npoint = x.size(2)

        # (bs, 9, npoint) -> (bs, 9*2, npoint, k)
        #print("1 ", x.shape)
        x = get_graph_feature(x, k=self.k)
        #print("2 ", x.shape)
        # (bs, 9*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv1(x)
        #print("3 ", x.shape)
        # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv2(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x1 = x.max(dim=-1, keepdim=False)[0]
        # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
        x = get_graph_feature(x1, k=self.k)
        # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv3(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv4(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
        x = get_graph_feature(x2, k=self.k)
        # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv5(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)      # (bs, 64*3, npoint)

        # (bs, 64*3, npoint) -> (bs, emb_dims, npoint)
        x = self.conv6(x)
        # (bs, emb_dims, npoint) -> (bs, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, npoint)          # (bs, 1024, npoint)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (bs, 1024+64*3, npoint)

        # (bs, 1024+64*3, npoint) -> (bs, 512, npoint)
        x = self.conv7(x)
        # (bs, 512, npoint) -> (bs, 256, npoint)
        x = self.conv8(x)
        x = self.dp1(x)
        # (bs, 256, npoint) -> (bs, 13, npoint)
        x = self.conv9(x)
        # (bs, 13, npoint) -> (bs, npoint, 13)
        x = x.transpose(2, 1).contiguous()
        

        return x
