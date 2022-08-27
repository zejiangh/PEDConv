# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
import time
from torch.autograd import Variable


__all__ = ["build_resnet", "resnet_versions", "resnet_configs", "resnet50", "resnet18", "resnet50d"]


class BatchConv2DLayer(nn.Module): # with option to include group convolutions
    def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1):
        super(BatchConv2DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h, w = x.shape
        b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        weight = weight.view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

        out = F.conv2d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i*self.groups, padding=self.padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])

        out = out.permute([1, 0, 2, 3, 4])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

        return out


class Attn4d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, groups, reduction=16):
        super(Attn4d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_share = nn.Linear(in_planes, in_planes // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc_inp = nn.Linear(in_planes // reduction, in_planes//groups)
        self.fc_oup = nn.Linear(in_planes // reduction, out_planes)
        self.fc_k   = nn.Linear(in_planes // reduction, kernel**2)
        self.sigmoid = nn.Sigmoid()
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.kernel_size = kernel
        self.groups = groups

    def forward(self, x):
        x = self.avg_pool(x).view(x.shape[0], -1)
        x = self.relu(self.fc_share(x))
        attn_inp = self.sigmoid(self.fc_inp(x)) # batchxCi
        attn_oup = self.sigmoid(self.fc_oup(x)) # batchxCo
        attn_k = self.sigmoid(self.fc_k(x)) # batchxkk
        attn = torch.bmm(attn_oup.unsqueeze(2), attn_inp.unsqueeze(1)).view(x.shape[0], -1)
        attn = torch.bmm(attn.unsqueeze(2), attn_k.unsqueeze(1)).view(x.shape[0], self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        return attn


class Attn4d_Tucker(nn.Module): # NO core tensor
    def __init__(self, in_planes, out_planes, kernel, groups, reduction=16, r=1):
        super(Attn4d_Tucker, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_share = nn.Linear(in_planes, in_planes // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc_inp = nn.Linear(in_planes // reduction, (in_planes//groups)*r)
        self.fc_oup = nn.Linear(in_planes // reduction, (out_planes)*r)
        self.fc_k   = nn.Linear(in_planes // reduction, (kernel**2)*r)
        self.sigmoid = nn.Sigmoid()
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.kernel_size = kernel
        self.groups = groups
        self.r = r

    def forward(self, x):
        x = self.avg_pool(x).view(x.shape[0], -1)
        x = self.relu(self.fc_share(x))
        if self.r == 1:
            attn_inp = self.sigmoid(self.fc_inp(x)) # batchxCi
            attn_oup = self.sigmoid(self.fc_oup(x)) # batchxCo
            attn_k = self.sigmoid(self.fc_k(x)) # batchxkk
            attn = torch.bmm(attn_oup.unsqueeze(2), attn_inp.unsqueeze(1)).view(x.shape[0], -1)
            attn = torch.bmm(attn.unsqueeze(2), attn_k.unsqueeze(1)).view(x.shape[0], self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        else:
            attn_inp = self.sigmoid(self.fc_inp(x)).view(x.shape[0], self.in_planes//self.groups, self.r) # batchxCixr
            attn_oup = self.sigmoid(self.fc_oup(x)).view(x.shape[0], self.out_planes, self.r) # batchxCoxr
            attn_k = self.sigmoid(self.fc_k(x)).view(x.shape[0], self.kernel_size**2, self.r) # batchxkkxr
            attn = 0
            for i, j, k in zip(np.arange(self.r), np.arange(self.r), np.arange(self.r)):
                temp = torch.bmm(attn_oup[:,:,i].unsqueeze(2), attn_inp[:,:,j].unsqueeze(1)).view(x.shape[0], -1)
                temp = torch.bmm(temp.unsqueeze(2), attn_k[:,:,k].unsqueeze(1)).view(x.shape[0], self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
                attn += temp
        return attn


class Attn4d_CP(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, groups, reduction=16, r=1):
        super(Attn4d_CP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_share = nn.Linear(in_planes, in_planes // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc_inp = nn.Linear(in_planes // reduction, (in_planes//groups)*r)
        self.fc_oup = nn.Linear(in_planes // reduction, (out_planes)*r)
        self.fc_k   = nn.Linear(in_planes // reduction, (kernel**2)*r)
        self.sigmoid = nn.Sigmoid()
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.kernel_size = kernel
        self.groups = groups
        self.r = r

    def forward(self, x):
        x = self.avg_pool(x).view(x.shape[0], -1)
        x = self.relu(self.fc_share(x))
        if self.r == 1:
            attn_inp = self.sigmoid(self.fc_inp(x)) # batchxCi
            attn_oup = self.sigmoid(self.fc_oup(x)) # batchxCo
            attn_k = self.sigmoid(self.fc_k(x)) # batchxkk
            attn = torch.bmm(attn_oup.unsqueeze(2), attn_inp.unsqueeze(1)).view(x.shape[0], -1)
            attn = torch.bmm(attn.unsqueeze(2), attn_k.unsqueeze(1)).view(x.shape[0], self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        else:
            attn_inp = self.sigmoid(self.fc_inp(x)).view(x.shape[0], self.in_planes//self.groups, self.r) # batchxCixr
            attn_oup = self.sigmoid(self.fc_oup(x)).view(x.shape[0], self.out_planes, self.r) # batchxCoxr
            attn_k = self.sigmoid(self.fc_k(x)).view(x.shape[0], self.kernel_size**2, self.r) # batchxkkxr
            attn = 0
            for i in range(self.r):
                temp = torch.bmm(attn_oup[:,:,i].unsqueeze(2), attn_inp[:,:,i].unsqueeze(1)).view(x.shape[0], -1)
                temp = torch.bmm(temp.unsqueeze(2), attn_k[:,:,i].unsqueeze(1)).view(x.shape[0], self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
                attn += temp
        return attn


class DYConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, prune=False):
        super(DYConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.prune = prune 
        self.ratio = 0. # default: no pruning

        self.attn = Attn4d(in_planes, out_planes, kernel_size, groups)
        self.conv = BatchConv2DLayer(in_planes, out_planes, stride, padding, dilation, groups)
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def weight_norm(self, weight):
        weight_copy = weight.abs().detach() # batchxCxCxkxk
        norm = torch.sum(weight_copy, dim=(2,3,4))
        return norm.cpu().numpy()

    def get_mask(self, weight, ratio):
        norm = self.weight_norm(weight) # batchxout_channels
        batch_mask = []
        for i in range(weight.shape[0]):
            mask = torch.ones(weight.shape[1]) # out_channels
            num_prune = int(ratio*weight.shape[1]) # number of pruned channels
            rank = np.argsort(norm[i]) # order from smallest to largest
            prune_idx = rank[:num_prune].tolist()
            mask[prune_idx] = 0
            batch_mask.append(mask)
        return torch.stack(batch_mask).cuda()

    def forward(self, x):
        attention = self.attn(x)
        weight = self.weight.unsqueeze(0) * attention # batchxCxCxkxk

        if self.prune and self.ratio != 0:
            mask = self.get_mask(weight, self.ratio).view(x.shape[0], self.out_planes, 1,1,1)
            weight.mul_(mask)

        out = self.conv(x.unsqueeze(1), weight).squeeze(1)
        return out


class Variational_Attn4d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, groups, reduction=16):
        super(Variational_Attn4d, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_share = nn.Linear(in_planes, in_planes // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc_inp = nn.Linear(in_planes // reduction, in_planes//groups)
        self.fc_oup = nn.Linear(in_planes // reduction, out_planes)
        self.fc_k   = nn.Linear(in_planes // reduction, kernel**2)
        self.sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(in_planes//groups+out_planes+kernel**2, in_planes//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes//reduction, in_planes),
            )

        self.out_planes = out_planes
        self.in_planes = in_planes
        self.kernel_size = kernel
        self.groups = groups

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def reparameterize(self, mu, logvar=None): # unit variance or learnable variance
        if self.training:
            if logvar != None:
                std = logvar.mul(0.5).exp_()
            else:
                std = 0.1
            esp = self.to_var(torch.randn(*mu.size()))
            z = mu + std * esp
        else:
            z = mu
        return z

    def info_max(self, xp, xp_): # assume Gaussian generative
        return F.mse_loss(xp, xp_)

    def forward(self, x):
        # contextual block
        x = self.avg_pool(x).view(x.shape[0], -1)

        # weight generation
        hidden = self.relu(self.fc_share(x))
        mu_inp = self.fc_inp(hidden) # batchxCi
        mu_oup = self.fc_oup(hidden) # batchxCo
        mu_k = self.fc_k(hidden) # batchxkk
        sample_inp = self.reparameterize(mu_inp)
        sample_oup = self.reparameterize(mu_oup)
        sample_k = self.reparameterize(mu_k)
        attn_inp = self.sigmoid(sample_inp)
        attn_oup = self.sigmoid(sample_oup)
        attn_k = self.sigmoid(sample_k)
        attn = torch.bmm(attn_oup.unsqueeze(2), attn_inp.unsqueeze(1)).view(x.shape[0], -1)
        attn = torch.bmm(attn.unsqueeze(2), attn_k.unsqueeze(1)).view(x.shape[0], self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        
        # reconstruction
        if self.training:
            reconstruct = self.decoder(torch.cat((sample_inp, sample_oup, sample_k), dim=1))
            self.IMLoss = self.info_max(x.detach(), reconstruct)
        else:
            self.IMLoss = 0
        
        return attn


class Variational_DYConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Variational_DYConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.attn = Variational_Attn4d(in_planes, out_planes, kernel_size, groups)
        self.conv = BatchConv2DLayer(in_planes, out_planes, stride, padding, dilation, groups)
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attention = self.attn(x)
        weight = self.weight.unsqueeze(0) * attention # batchxCxCxkxk
        out = self.conv(x.unsqueeze(1), weight).squeeze(1)
        return out


# ================================================================================================
'''
ResNet50
'''

class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Variational_DYConv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = Variational_DYConv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = Variational_DYConv2d(cfg[2], planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet50d(nn.Module):
    def __init__(self, wmul=1.0):
        super(resnet50d, self).__init__()
        block = Bottleneck
        s1 = min(int(wmul*64)+1, 64)
        s2 = min(int(wmul*128)+1, 128)
        s3 = min(int(wmul*256)+1, 256)
        s4 = min(int(wmul*512)+1, 512)
        cfg = [[64, s1, s1], [256, s1, s1] * 2, [256, s2, s2], [512, s2, s2] * 3, [512, s3, s3], [1024, s3, s3] * 5, [1024, s4, s4], [2048, s4, s4] * 2, [2048]]
        cfg = [item for sub_list in cfg for item in sub_list]
        assert len(cfg) == 49, "Length of cfg_official is not right"
        self.cfg=cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, cfg[0:9], 256, 3, stride=2)
        self.layer2 = self._make_layer(block, cfg[9:21], 512, 4, stride=2)
        self.layer3 = self._make_layer(block, cfg[21:39], 1024, 6, stride=2)
        self.layer4 = self._make_layer(block, cfg[39:48], 2048, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], 1000)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class resnet50(nn.Module):
    def __init__(self, wmul=1.0):
        super(resnet50, self).__init__()
        block = Bottleneck
        s1 = min(int(wmul*64)+1, 64)
        s2 = min(int(wmul*128)+1, 128)
        s3 = min(int(wmul*256)+1, 256)
        s4 = min(int(wmul*512)+1, 512)
        cfg = [[64, s1, s1], [256, s1, s1] * 2, [256, s2, s2], [512, s2, s2] * 3, [512, s3, s3], [1024, s3, s3] * 5, [1024, s4, s4], [2048, s4, s4] * 2, [2048]]
        cfg = [item for sub_list in cfg for item in sub_list]
        assert len(cfg) == 49, "Length of cfg_official is not right"
        self.cfg=cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[0:9], 256, 3)
        self.layer2 = self._make_layer(block, cfg[9:21], 512, 4, stride=2)
        self.layer3 = self._make_layer(block, cfg[21:39], 1024, 6, stride=2)
        self.layer4 = self._make_layer(block, cfg[39:48], 2048, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], 1000)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)]))

        return nn.Sequential(*layers)

    def gather_BN(self, sparsity):
        l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
        l2 = (np.asarray(l1)+1).tolist()
        l3 = (np.asarray(l2)+1).tolist()
        skip = [5,15,28,47]
        total = 0
        bn_count = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_count in l1 + l2:
                    total += m.weight.data.shape[0]
                    bn_count += 1
                    continue
                bn_count += 1
        bn = torch.zeros(total)
        index = 0
        bn_count = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_count in l1 + l2:
                    size = m.weight.data.shape[0]
                    bn[index:(index+size)] = m.weight.data.abs().clone()
                    index += size
                    bn_count += 1
                    continue
                bn_count += 1
        y, i = torch.sort(bn)
        thre_index = int(total * sparsity)
        thre = y[thre_index]
        prune_ratio = []
        bn_count = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_count in l1 + l2:
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre).float().cuda()
                    prune_ratio.append((mask.shape[0] - torch.sum(mask).item()) / mask.shape[0])
                    bn_count += 1
                    continue
                bn_count += 1
        return prune_ratio

    def cal_FLOPs(self, r):
        '''
        r: prune ratio; list of length = 32
        '''
        cfgs = [
            # inp, mid, oup, res, inp_ratio, mid_ratio
            [64, 64, 256, 56, r[0], r[1]],
            [256, 64, 256, 56, r[2], r[3]],
            [256, 64, 256, 56, r[4], r[5]],
            [256, 128, 512, 28, r[6], r[7]],
            [512, 128, 512, 28, r[8], r[9]],
            [512, 128, 512, 28, r[10], r[11]],
            [512, 128, 512, 28, r[12], r[13]],
            [512, 256, 1024, 14, r[14], r[15]],
            [1024, 256, 1024, 14, r[16], r[17]],
            [1024, 256, 1024, 14, r[18], r[19]],
            [1024, 256, 1024, 14, r[20], r[21]],
            [1024, 256, 1024, 14, r[22], r[23]],
            [1024, 256, 1024, 14, r[24], r[25]],
            [1024, 512, 2048, 7, r[26], r[27]],
            [2048, 512, 2048, 7, r[28], r[29]],
            [2048, 512, 2048, 7, r[30], r[31]],
        ]
        FLOPs = 3 * 64 * 7**2 * 112**2 / 1e9 # input convolution
        for inp, mid, oup, res, inp_ratio, mid_ratio in cfgs:
            conv1 = 1**2 * res**2 * inp * int((1-inp_ratio)*mid) / 1e9
            conv2 = 3**2 * res**2 * int((1-inp_ratio)*mid) * int((1-mid_ratio)*mid) / 1e9
            conv3 = 1**2 * res**2 * int((1-mid_ratio)*mid) * oup / 1e9
            FLOPs += conv1 + conv2 + conv3
        FLOPs += (2048 * 1000 + 1000) / 1e9 # FC
        return FLOPs

    def guided_pruning(self, prune_ratio):
        stage1 = sum(prune_ratio[:6]) / 6
        stage2 = sum(prune_ratio[6:14]) / 8
        stage3 = sum(prune_ratio[14:26]) / 12
        stage4 = sum(prune_ratio[26:]) / 6
        prune_ratio = [stage1] * 6 + [stage2] * 8 + [stage3] * 12 + [stage4] * 6
        return prune_ratio

    def smart_ratio(self, max_flops):
        s_min = 0
        s_max = 1
        for _ in range(1000):
            sparsity = 0.5 * (s_min + s_max)
            prune_ratio = self.gather_BN(sparsity)
            prune_ratio = self.guided_pruning(prune_ratio)
            FLOPs = self.cal_FLOPs(prune_ratio)
            if np.abs(FLOPs - max_flops) / max_flops <= 1e-3:
                return prune_ratio, FLOPs
            else:
                if FLOPs < max_flops:
                    s_max = sparsity
                else:
                    s_min = sparsity
        return prune_ratio, FLOPs

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ================================================================================================  
'''
ResNet18
'''

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Variational_DYConv2d(inplanes, cfg[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Variational_DYConv2d(cfg[0], planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class resnet18(nn.Module):
    def __init__(self, width_ratio=1):
        self.width_ratio = width_ratio
        self.inplanes = int(64*self.width_ratio)
        super(resnet18, self).__init__()
        block = BasicBlock
        cfg = [64,64,128,128,256,256,512,512,512]
        cfg = [int(element * self.width_ratio) for element in cfg]
        self.cfg=cfg
        self.conv1 = nn.Conv2d(3, int(64*self.width_ratio), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*self.width_ratio))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[0:2], int(64*self.width_ratio), 2)
        self.layer2 = self._make_layer(block, cfg[2:4], int(128*self.width_ratio), 2, stride=2)
        self.layer3 = self._make_layer(block, cfg[4:6], int(256*self.width_ratio), 2, stride=2)
        self.layer4 = self._make_layer(block, cfg[6:8], int(512*self.width_ratio), 2, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_() 
            
    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, cfg[:1], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[1*i:1*(i+1)]))
        return nn.Sequential(*layers)

    def gather_BN(self, sparsity):
        l1 = [2,4, 6,9, 11,14, 16,19]
        l2 = (np.asarray(l1)+1).tolist()
        skip = [8,13,18]
        total = 0
        bn_count = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_count in l1 + l2:
                    total += m.weight.data.shape[0]
                    bn_count += 1
                    continue
                bn_count += 1
        bn = torch.zeros(total)
        index = 0
        bn_count = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_count in l1 + l2:
                    size = m.weight.data.shape[0]
                    bn[index:(index+size)] = m.weight.data.abs().clone()
                    index += size
                    bn_count += 1
                    continue
                bn_count += 1
        y, i = torch.sort(bn)
        thre_index = int(total * sparsity)
        thre = y[thre_index]
        prune_ratio = []
        bn_count = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_count in l1 + l2:
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre).float().cuda()
                    prune_ratio.append((mask.shape[0] - torch.sum(mask).item()) / mask.shape[0])
                    bn_count += 1
                    continue
                bn_count += 1
        return prune_ratio

    def cal_FLOPs(self, r):
        '''
        r: prune ratio; list of length = 16
        '''
        cfgs = [
            # inp, oup, k, res, inp_ratio, oup_ratio
            [3, 64, 7, 112, 0, 0],
            [64, 64, 3, 56, 0, r[0]],
            [64, 64, 3, 56, r[0], r[1]],
            [64, 64, 3, 56, 0, r[2]],
            [64, 64, 3, 56, r[2], r[3]],
            [64, 128, 3, 28, 0, r[4]],
            [128, 128, 3, 28, r[4], r[5]],
            [128, 128, 3, 28, 0, r[6]],
            [128, 128, 3, 28, r[6], r[7]],
            [128, 256, 3, 14, 0, r[8]],
            [256, 256, 3, 14, r[8], r[9]],
            [256, 256, 3, 14, 0, r[10]],
            [256, 256, 3, 14, r[10], r[11]],
            [256, 512, 3, 7, 0, r[12]],
            [512, 512, 3, 7, r[12], r[13]],
            [512, 512, 3, 7, 0, r[14]],
            [512, 512, 3, 7, r[14], r[15]],
        ]
        FLOPs = 0
        for inp, oup, k, res, inp_ratio, oup_ratio in cfgs:
            FLOPs += k**2 * res**2 * int((1-inp_ratio)*inp) * int((1-oup_ratio)*oup) / 1e9
        return FLOPs

    def guided_pruning(self, prune_ratio):
        stage1 = sum(prune_ratio[:4]) / 4
        stage2 = sum(prune_ratio[4:8]) / 4
        stage3 = sum(prune_ratio[8:12]) / 4
        stage4 = sum(prune_ratio[12:]) / 4
        prune_ratio = [stage1] *4 + [stage2] * 4 + [stage3] * 4 + [stage4] * 4
        return prune_ratio

    def smart_ratio(self, max_flops):
        s_min = 0
        s_max = 1
        for _ in range(1000):
            sparsity = 0.5 * (s_min + s_max)
            prune_ratio = self.gather_BN(sparsity)
            prune_ratio = self.guided_pruning(prune_ratio)
            FLOPs = self.cal_FLOPs(prune_ratio)
            if np.abs(FLOPs - max_flops) / max_flops <= 1e-3:
                return prune_ratio, FLOPs
            else:
                if FLOPs < max_flops:
                    s_max = sparsity
                else:
                    s_min = sparsity
        return prune_ratio, FLOPs

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
          
# ================================================================================================

resnet_configs = {
    "classic": {
        "conv": nn.Conv2d,
        "conv_init": "fan_out",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "fanin": {
        "conv": nn.Conv2d,
        "conv_init": "fan_in",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "grp-fanin": {
        "conv": nn.Conv2d,
        "conv_init": "fan_in",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "grp-fanout": {
        "conv": nn.Conv2d,
        "conv_init": "fan_out",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
}

resnet_versions = {
    "resnet50": {
        "net": resnet50,
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
    "resnet18":{
        "net": resnet18,
        "block": BasicBlock,
        "layers": [2,2,2,2],
        "widths": [64, 128, 256, 512],
    },
    "resnet50d": {
        "net": resnet50d,
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
}

def build_resnet(version, config, num_classes, cfg=None, verbose=True):
    print('>============================== SELF-DEFINED MODELS ====================================<')
    if version == "resnet50":
        model = resnet50(cfg)
    elif version == 'resnet18':
        model = resnet18(cfg)
    elif version == 'resnet50d':
        model = resnet50d(cfg)
    else:
        print('no model found')
        sys.exit()
    version = resnet_versions[version]
    config = resnet_configs[config]
    if verbose:
        print("Version: {}".format(version))
        print("Config: {}".format(config))
        print("Num classes: {}".format(num_classes))

    return model
