import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from logone.utilities.utils import zoom_to_bounding_box, normalize_logo

class UNet(nn.Module):
    def __init__(self, in_channels, height, width):
        super(UNet, self).__init__()
        self.h_ = height
        self.w_ = width
        out_features = 9  # Set the number of output parameters

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder1 = self.conv_block(1024, 512)
        self.decoder2 = self.conv_block(512, 256)
        self.decoder3 = self.conv_block(256, 128)
        self.decoder4 = self.conv_block(128, 64)

        # Flattening and Fully Connected Layers
        # The size after the last decoding step needs to be calculated according to your input size
        # Assuming the input dimensions halve at each down-sampling step in the encoder
        final_dim = (height // 16) * (width // 16) * 64  # adjust depending on your specific architecture and input size
        self.fc1 = nn.Linear(final_dim, 512)
        self.fc2 = nn.Linear(512, out_features)  # Output layer with 9 units

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoding
        x = x.view(-1,6,self.h_, self.w_)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoding + Concatenation (skip connections)
        dec1 = self.decoder1(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False))
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec2 = self.decoder2(F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=False))
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec3 = self.decoder3(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec4 = self.decoder4(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False))
        dec4 = torch.cat((dec4, enc1), dim=1)

        # Flatten and fully connected layers
        flattened = torch.flatten(dec4, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        final_output = self.fc2(fc1_output)

        return final_output

class UnPyramidCNN(nn.Module):
    def __init__(self, input_channels, output_classes, h, w, kernel_size=7):
        super(UnPyramidCNN, self).__init__()
        self.h_ = h
        self.w_ = w
        self.ch1 = 128
        self.ch2 = 64
        self.ch3 = 32
        self.flat_dim = self.ch3*h*w

        pad = kernel_size//2

        # self.flat_dim = (self.pc*3) * h * w
        # self.flat_dim = 32*32*3

        # self.conv11 = nn.Conv2d(input_channels, self.pc, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(input_channels, self.ch1, kernel_size=kernel_size, padding=pad)
        self.conv11 = nn.Conv2d(self.ch1, self.ch1, kernel_size=kernel_size, padding=pad)
        # self.conv22 = nn.Conv2d(3, self.pc, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.ch1, self.ch2, kernel_size=kernel_size, padding=pad)
        self.conv22 = nn.Conv2d(self.ch2, self.ch2, kernel_size=kernel_size, padding=pad)
        # self.conv33 = nn.Conv2d(3, self.pc, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.ch2, self.ch3, kernel_size=kernel_size, padding=pad)
        self.conv33 = nn.Conv2d(self.ch3, self.ch3, kernel_size=kernel_size, padding=pad)
        # self.conv44 = nn.Conv2d(3, self.pc, kernel_size=7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU()
        )

        self.fclast = nn.Linear(128, output_classes)


    def forward(self, x):
        x = x.view(-1,256,self.h_, self.w_)
        # x = self.pool(F.relu(self.conv1(x)))  # Reduce dimensions
        # x = self.pool(F.relu(self.conv2(x)))  # Reduce dimensions
        # x = self.pool(F.relu(self.conv3(x)))  # Reduce dimensions
        x = F.relu(self.conv11(F.relu(self.conv1(x))))  # Reduce dimensions
        x = F.relu(self.conv22(F.relu(self.conv2(x))))  # Reduce dimensions
        x = F.relu(self.conv33(self.conv3(x)))  # Reduce dimensions

        x = x.view(-1,self.flat_dim)
        x = self.fc1(x)
        x = self.fclast(x)
        return x

class PyramidCNN(nn.Module):
    def __init__(self, input_channels, output_classes, h, w, kernel_size=7):
        super(PyramidCNN, self).__init__()
        self.h_ = h
        self.w_ = w
        self.ch1 = 8
        self.ch2 = 16
        self.ch3 = 32
        self.flat_dim1 = h//2*w//2*self.ch1
        self.flat_dim2 = h//4*w//4*self.ch2
        self.flat_dim3 = h//8*w//8*self.ch3
        # print(self.flat_dim1, self.flat_dim2, self.flat_dim3)

        pad = kernel_size//2

        # self.flat_dim = (self.pc*3) * h * w
        # self.flat_dim = 32*32*3

        # self.conv11 = nn.Conv2d(input_channels, self.pc, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(input_channels, self.ch1, kernel_size=kernel_size, padding=pad)
        # self.conv22 = nn.Conv2d(3, self.pc, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.ch1, self.ch2, kernel_size=kernel_size, padding=pad)
        # self.conv33 = nn.Conv2d(3, self.pc, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.ch2, self.ch3, kernel_size=kernel_size, padding=pad)
        # self.conv44 = nn.Conv2d(3, self.pc, kernel_size=7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_dim1, 128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.flat_dim2, 128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.flat_dim3, 128),
            nn.ReLU()
        )

        self.fclast = nn.Linear(128, output_classes)


    def forward(self, x):
        # x1 = F.relu(self.conv11(x)) 
        x = x.view(-1,6,self.h_, self.w_)
        x = self.pool(F.relu(self.conv1(x)))  # Reduce dimensions
        x1 = self.fc1(x.view(-1, self.flat_dim1))

        # x2 = F.relu(self.conv22(x))
        x = self.pool(F.relu(self.conv2(x)))  # Reduce dimensions
        x2 = self.fc2(x.view(-1, self.flat_dim2))

        # x3 = F.relu(self.conv33(x))
        x = self.pool(F.relu(self.conv3(x)))  # Reduce dimensions
        x3 = self.fc3(x.view(-1, self.flat_dim3))

        # x4 = F.relu(self.conv44(x))

        # x = torch.cat([x1,x2,x3], dim=1)
        x = self.fclast(x1+x2+x3)
        return x