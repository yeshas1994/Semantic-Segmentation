import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function
from models.backbones.resnet import get_resnet_pretrained 
import logging

def bilinear_init(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class FCN32(nn.Module):

    def __init__(self, backbone_name, in_channel=512, num_classes=11):
        super(FCN32, self).__init__()
        self.backbone = get_resnet_pretrained(model_name=backbone_name)
        self.cls_num = num_classes
        self.relu    = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.cls_num, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.cls_num)
        self.DCN32 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=64, stride=32, dilation=1, padding=16)
        self.DCN32.weight.data = bilinear_init(self.cls_num, self.cls_num, 64)
        self.dbn32 = nn.BatchNorm2d(self.cls_num)


    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x = self.bn1(self.relu(self.Conv1x1(x4)))
        x = self.dbn32(self.relu(self.DCN32(x)))
        return x

class FCN16(nn.Module):

    def __init__(self, backbone_name, in_channel=512, num_classes=11):
        super(FCN16, self).__init__()
        self.backbone = get_resnet_pretrained(model_name=backbone_name)
        logging.info(self.backbone)
        self.cls_num = num_classes
        self.relu    = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.cls_num, kernel_size=1)
        self.Conv1x1_x4 = nn.Conv2d(int(in_channel/2), self.cls_num, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.cls_num)
        self.DCN2 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN2.weight.data = bilinear_init(self.cls_num, self.cls_num, 4)
        self.dbn2 = nn.BatchNorm2d(self.cls_num)
        self.DCN16 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=32, stride=16, dilation=1, padding=8)
        self.DCN16.weight.data = bilinear_init(self.cls_num, self.cls_num, 32)
        self.dbn16 = nn.BatchNorm2d(self.cls_num)


    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x = self.bn1(self.relu(self.Conv1x1(x4)))
        x3 = self.bn1(self.relu(self.Conv1x1_x4(x3)))
        x = self.dbn2(self.relu(self.DCN2(x)))
        x = x + x3
        x = self.dbn16(self.relu(self.DCN16(x)))

        return x

class FCN8(nn.Module):

    def __init__(self, backbone_name, in_channel=512, num_classes=11):
        super(FCN8, self).__init__()
        self.backbone = get_resnet_pretrained(model_name=backbone_name)
        self.cls_num = num_classes
        self.relu    = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.cls_num, kernel_size=1)
        self.Conv1x1_x4 = nn.Conv2d(int(in_channel/2), self.cls_num, kernel_size=1)
        self.Conv1x1_x3 = nn.Conv2d(int(in_channel/4), self.cls_num, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.cls_num)
        self.DCN2 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN2.weight.data = bilinear_init(self.cls_num, self.cls_num, 4)
        self.dbn2 = nn.BatchNorm2d(self.cls_num)
        self.DCN4 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN4.weight.data = bilinear_init(self.cls_num, self.cls_num, 4)
        self.dbn4 = nn.BatchNorm2d(self.cls_num)
        self.DCN8 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=16, stride=8, dilation=1, padding=4)
        self.DCN8.weight.data = bilinear_init(self.cls_num, self.cls_num, 16)
        self.dbn8 = nn.BatchNorm2d(self.cls_num)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x = self.bn1(self.relu(self.Conv1x1(x4)))
        x3 = self.bn1(self.relu(self.Conv1x1_x4(x3)))
        x = self.dbn2(self.relu(self.DCN2(x)))
        x = x + x3
        x2 = self.bn1(self.relu(self.Conv1x1_x3(x2)))
        x = self.dbn4(self.relu(self.DCN4(x)))
        x = x + x2
        x = self.dbn8(self.relu(self.DCN8(x)))

        return x
