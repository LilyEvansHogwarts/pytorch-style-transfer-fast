import torch
import torchvision
import os
import numpy as np
from collections import namedtuple
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Vgg16(torch.nn.Module):
    def __init__(self, content_indices=[8], style_indices=[3, 8, 15, 22], requires_grad=True, show_progress=False):
        super(Vgg16, self).__init__()
        self.content_indices = content_indices
        self.style_indices = style_indices 

        self.features = sorted(set(content_indices + style_indices))
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True, progress=show_progress).features

        start_index = 0
        for i in range(len(self.features)):
            layer = torch.nn.Sequential()
            while start_index <= self.features[i]:
                layer.add_module(str(start_index), vgg_pretrained_features[start_index])
                start_index += 1

            setattr(self, 'slice'+str(i+1), layer)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        result = inputs
        content_features = []
        style_features = []

        for i in range(len(self.features)):
            layer = getattr(self, 'slice'+str(i+1))
            result = layer(result)

            if self.features[i] in self.content_indices:
                content_features.append(result)

            if self.features[i] in self.style_indices:
                style_features.append(gram_matrix(result))

        vgg_outputs = namedtuple('VggOutputs', ['content_features', 'style_features'])
        outputs = vgg_outputs(content_features, style_features)

        return outputs

PerceptualLossNet = Vgg16



class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.relu = torch.nn.ReLU()

        num_of_channels = [3, 32, 64, 128]
        kernel_sizes = [9, 3, 3]
        stride_sizes = [1, 2, 2]
        self.conv1 = torch.nn.Conv2d(num_of_channels[0], num_of_channels[1], kernel_sizes[0], stride_sizes[0], padding=kernel_sizes[0]//2, padding_mode='reflect')
        self.in1 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.conv2 = torch.nn.Conv2d(num_of_channels[1], num_of_channels[2], kernel_sizes[1], stride_sizes[1], padding=kernel_sizes[1]//2, padding_mode='reflect')
        self.in2 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.conv3 = torch.nn.Conv2d(num_of_channels[2], num_of_channels[3], kernel_sizes[2], stride_sizes[2], padding=kernel_sizes[2]//2, padding_mode='reflect')
        self.in3 = torch.nn.InstanceNorm2d(num_of_channels[3], affine=True)

        self.res1 = ResidualBlock(num_of_channels[3])
        self.res2 = ResidualBlock(num_of_channels[3])
        self.res3 = ResidualBlock(num_of_channels[3])
        self.res4 = ResidualBlock(num_of_channels[3])
        self.res5 = ResidualBlock(num_of_channels[3])

        num_of_channels.reverse()
        kernel_sizes.reverse()
        stride_sizes.reverse()
        self.up1 = UpsampleConvLayer(num_of_channels[0], num_of_channels[1], kernel_sizes[0], stride_sizes[0])
        self.in4 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.up2 = UpsampleConvLayer(num_of_channels[1], num_of_channels[2], kernel_sizes[1], stride_sizes[1])
        self.in5 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.up3 = torch.nn.Conv2d(num_of_channels[2], num_of_channels[3], kernel_sizes[2], stride_sizes[2], padding=kernel_sizes[2]//2, padding_mode='reflect')

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))
        return self.up3(y)

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.upsampling_factor = stride
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        stride_size = 1
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, stride_size, padding=kernel_size//2, padding_mode='reflect')
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, stride_size, padding=kernel_size//2, padding_mode='reflect')
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv3 = torch.nn.Conv2d(channels, channels, kernel_size, stride_size, padding=kernel_size//2, padding_mode='reflect')
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

