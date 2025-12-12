#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: naraysa & akshitac8
# 原来biam+全连接层+scb+rcb
# tr2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
import random
import torchvision
import ssl
from transformer import TransformerEncoder
from typing import Optional, Tuple, Union, Dict

ssl._create_default_https_context = ssl._create_unverified_context

random.seed(3483)
np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)


class CONV1_1(nn.Module):
    def __init__(self, num_in, num_out, kernel=1):
        super(CONV1_1, self).__init__()
        self.body = nn.Conv2d(num_in, num_out, kernel, padding=int((kernel - 1) / 2), dilation=1)

    def forward(self, x):
        x = self.body(x)
        return x


class CONV3_3(nn.Module):
    def __init__(self, num_in, num_out, kernel=3):
        super(CONV3_3, self).__init__()
        self.body = nn.Conv2d(num_in, num_out, kernel, padding=int((kernel - 1) / 2), dilation=1)
        self.bn = nn.BatchNorm2d(num_out, affine=True, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.body(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class C_R(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=True)
        # 第一个卷积层将输入通道数设置为d_model，输出通道数设置为d_ff，核大小为1
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=True)
        # 第二个卷积层将输入通道数设置为d_ff，输出通道数设置为d_model，核大小为1

    def forward(self, x):
        x_out = self.conv2(F.relu(self.conv1(x), True))
        return x_out


class GoogLeNetFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.features = pretrained_model

    def forward(self, x):
        # 以下代码基于GoogLeNet的forward方法的实际实现
        # 你可能需要根据实际的网络结构进行调整
        x = self.features.conv1(x)
        x = self.features.maxpool1(x)
        x = self.features.conv2(x)
        x = self.features.conv3(x)
        x = self.features.maxpool2(x)
        x = self.features.inception3a(x)
        x = self.features.inception3b(x)
        x = self.features.maxpool3(x)
        x = self.features.inception4a(x)
        # 假设取inception4a这一层的输出作为特征图
        return x

    def forward_from_inception4a(self, x):
        # 继续前向传播
        x = self.features.inception4b(x)
        x = self.features.inception4c(x)
        x = self.features.inception4d(x)
        x = self.features.inception4e(x)
        x = self.features.maxpool4(x)
        x = self.features.inception5a(x)
        x = self.features.inception5b(x)
        # N x 1024 x 7 x 7
        global_features = self.features.avgpool(x)
        # N x 1024 x 1 x 1
        # 通过adaptive avg pool确保输出大小正确
        global_features = torch.flatten(global_features, start_dim=1)
        # N x 1024
        return global_features


class RCB(nn.Module):
    """
    Region contextualized block
    """

    def __init__(self, heads=8, d_model=512, d_ff=1024, dropout=0.1):
        super(RCB, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.w_q = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=True)
        self.w_k = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=True)
        self.w_v = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=True)
        self.w_o = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.sub_network = C_R(d_model, d_ff)

    def F_R(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(scores == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores

    def forward(self, q_feat, k_feat, v_feat):
        if k_feat is None:
            k_feat = q_feat
        bs = q_feat.size(0)
        spa = q_feat.size(-1)
        residual = q_feat
        k_h_r = self.w_k(k_feat).view(bs, self.h, self.d_k, spa * spa).transpose(3, 2)
        q_h_r = self.w_q(q_feat).view(bs, self.h, self.d_k, spa * spa).transpose(3, 2)
        v_h_r = self.w_v(v_feat).view(bs, self.h, self.d_k, spa * spa).transpose(3, 2)
        r_h = self.F_R(q_h_r, k_h_r, v_h_r, self.d_k, self.dropout_1)
        alpha_h = torch.matmul(r_h, v_h_r)
        o_r = alpha_h.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        o_r = o_r.permute(0, 2, 1)
        o_r = o_r.view(-1, self.d_model, spa, spa)
        o_r = self.dropout_2(self.w_o(o_r))
        o_r += residual
        input_o_r = o_r
        e_r = self.sub_network(o_r)
        e_r += input_o_r
        return e_r


class SCB(nn.Module):
    """
    scene contextualized block
    """

    def __init__(self):
        super(SCB, self).__init__()
        self.channel_dim = 256
        self.sigmoid = nn.Sigmoid()
        self.gcdropout = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.2, False)
        self.w_g = nn.Conv2d(in_channels=1024, out_channels=self.channel_dim, kernel_size=1,
                             bias=True)  # nn.Linear(4096, self.channel_dim, bias=False) #
        self.gcff = CONV3_3(num_in=self.channel_dim, num_out=self.channel_dim)
        self.channel_conv = CONV1_1(num_in=self.channel_dim, num_out=self.channel_dim)

    def F_G(self, q, k):
        r_g = q * k
        r_g = self.sigmoid(r_g)
        r_g = r_g.view(-1, self.channel_dim, 1)
        return r_g

    def forward(self, h_r, x_g):
        # import pdb;pdb.set_trace()
        q_g = self.lrelu(self.channel_conv(h_r))
        v_g = self.lrelu(self.channel_conv(h_r))
        k_g = self.w_g(self.gcdropout(x_g).view(-1, 1024, 1, 1))
        # k_g = self.w_g(self.gcdropout(x_g))
        q_g_value = q_g.view(-1, self.channel_dim, 196).mean(-1).repeat(1, 1, 1).view(-1, self.channel_dim)
        r_g = self.F_G(q_g_value, k_g.view(-1, self.channel_dim))
        # r_g = self.F_G(q_g_value,k_g)
        c_g = r_g.unsqueeze(3).unsqueeze(4) * v_g.unsqueeze(2)
        c_g = c_g.view(-1, self.channel_dim, 14, 14)
        e_g = c_g + self.gcff(c_g)
        return e_g


class BlockProcessor(nn.Module):
    def __init__(self):
        super(BlockProcessor, self).__init__()
        # 更改输出通道数为256
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # 使用转置卷积进行上采样，或者可以使用其他上采样方法
        self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        n_blocks, batch_size, channels, height, width = x.size()
        # 重塑x以便在所有块上并行应用卷积
        x = x.view(-1, channels, height, width)
        x = self.conv(x)  # 应用卷积层
        x = self.relu(x)  # 应用ReLU激活函数
        x = self.upsample(x)  # 上采样到14x14
        x = F.relu(x)  # 应用ReLU激活函数
        # 将x重新塑形为原始块的形状
        x = x.view(n_blocks, batch_size, -1, height * 2, width * 2)
        # 使用最大池化来融合特征
        x, _ = x.max(dim=0)
        return x


class BiAM(nn.Module):
    def __init__(self):
        super(BiAM, self).__init__()
        # D = dim_feature[1]  #### D is the feature dimension of attention windows
        self.channel_dim = 256
        self.conv_3X3 = CONV3_3(num_in=self.channel_dim, num_out=self.channel_dim)
        # self.region_context_block = CompactFeatureModel()
        self.region_context_block = RCB(heads=8, d_model=self.channel_dim, d_ff=self.channel_dim * 2, dropout=0.1)
        # self.scene_context_block = EnhancedGlobalFeatureExtractor()
        self.scene_context_block = SCB()
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))  # 创建一个自适应平均池化层。
        # self.W = nn.Linear(dim_w2v, D, bias=True)
        self.conv_1X1 = CONV1_1(num_in=self.channel_dim*2, num_out=self.channel_dim*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.classifier = nn.Sequential()  # 创建一个序列容器classifier。
        self.classifier.add_module(name="flatten", module=nn.Flatten())  # 向classifier中添加一个将输入展平的Flatten模块。
        self.classifier.add_module(name="fc",
                                   module=nn.Linear(in_features=512, out_features=636))  # 向classifier中添加一个全连接层。
        #self.adjuster = FeatureAdjuster()
        # self.feature_reshape = FeatureReshape(input_dim=4096, output_channels=256, output_height=14, output_width=14)
        self.block_processor = BlockProcessor()
        self.SelfAttention = SelfAttention(heads=8, d_model=self.channel_dim)


    def forward(self, images, googlenet):

        # 提取特征
        feature_map = googlenet.forward(images)  ## 利用googlenet得到浅层特征图 为局部特征   假设这里的尺寸是torch.Size([32, 512, 14, 14])
        #print(feature_map.size())#orch.Size([32, 512, 14, 14])
        local_features = split_feature_map(feature_map)  # 将局部特征切分成4个部分 torch.Size([4, 32, 512, 7, 7])
        x_r = self.block_processor(local_features)  # 对每一个局部特征进行处理并修改维度[32,256,14,14]  目前可以考虑修改的地方
        # 获取全局特征
        x_g = googlenet.forward_from_inception4a(feature_map)  #补充googlenet网络 得到全局特征 [32, 1024]
        #print(x_g.size())#torch.Size([32, 1024])
        h_r = self.conv_3X3(x_r)  # 对x_r进行卷积 32,256,14,14

        e_r = self.region_context_block(h_r, h_r, h_r)  # RCB# torch.Size([16, 256, 14, 14])
        # global_features现在的大小为(batch_size, num_features) 16*1024
        e_g = self.scene_context_block(h_r, x_g)  # torch.Size([16, 256, 14, 14])
        e_f = self.process_query_key_value(e_r,e_g)

        # 将空间注意力模块应用到输入数据上
        # print(combined_features.size())
        e_f = self.lrelu(self.conv_1X1(e_f))  # 过self.lrelu激活函数和self.conv_1X1卷积对e_f进行处理
        # e_f = self.conv11(e_f)  # 将x传入conv11序列中进行卷积操作。#32*512*14*14
        # 分类
        x = self.global_average_pool(e_f)  # 将x传入全局平均池化层中。#32*512*1*1
        logits = self.classifier(x)

        return logits

    # 对 k_g, v_g 和 q_r 进行操作
    def process_query_key_value(self,e_r,e_g):
        #处理e_r特征，使用e_g作为键和值
        q_r, k_g, v_g = self.SelfAttention.w_q(e_r), self.SelfAttention.w_k(e_g), self.SelfAttention.w_v(e_g)
        out_r = self.SelfAttention(q_r, k_g, v_g)  # 得到处理后的e_r特征

        # 处理e_g特征，使用e_r作为键和值
        q_g, k_r, v_r = self.SelfAttention.w_q(e_g), self.SelfAttention.w_k(e_r), self.SelfAttention.w_v(e_r)
        out_g = self.SelfAttention(q_g, k_r, v_r)  # 得到处理后的e_g特征

        # 融合两个特征（这里简单使用相加，也可以根据需要使用其他融合方法）
        fused_feature = torch.cat([out_r, out_g], dim=1) # 或者使用其他融合策略，如concat后卷积等。

        return fused_feature


def split_feature_map(feature_map):
    n_blocks = 2
    block_size_h = feature_map.shape[2] // n_blocks
    block_size_w = feature_map.shape[3] // n_blocks
    blocks = []

    for i in range(n_blocks):
        for j in range(n_blocks):
            block = feature_map[:, :, i * block_size_h:(i + 1) * block_size_h, j * block_size_w:(j + 1) * block_size_w]
            blocks.append(block.unsqueeze(0))  # Add an extra dimension for the blocks

    # 使用torch.cat在新的维度上拼接块
    return torch.cat(blocks, dim=0)
    # output_tensor的大小将是torch.Size([4, 32, 512, 7, 7])。这表示有4个不同的块，每个块的大小都是[32, 512, 7, 7]，并且它们都存储在同一个张量中

