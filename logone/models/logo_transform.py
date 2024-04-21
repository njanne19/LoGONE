import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from utils import zoom_to_bounding_box

class LogoTransformDataset(Dataset):
    def __init__(self,annotations_file, img_dir):
        self.img_paths = np.genfromtxt(annotations_file,
                                       delimiter=',', 
                                       dtype=str, 
                                       skip_header=1,
                                       usecols=(0,1))
        self.labels = np.genfromtxt(annotations_file,
                                    delimiter=',',
                                    dtype=float,
                                    skip_header=1,
                                    usecols=(2,3,4,5,6,7,8,9,10))
        self.img_dir = img_dir
        with open(annotations_file) as f:
            self.label_header = f.readline().strip('\n')

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        img_path_t = os.path.join(self.img_dir, self.img_paths[idx,0])
        img_path_tt = os.path.join(self.img_dir, self.img_paths[idx,1])
        conc_img = np.concatenate((zoom_to_bounding_box(cv2.imread(img_path_t)), zoom_to_bounding_box(cv2.imread(img_path_tt))), axis=2)
        return conc_img.astype(np.float32), self.labels[idx]




class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, width, height):
        super(UNet, self).__init__()

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
        
        # Final output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Linear(64*height*width)
        )
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = nn.functional.max_pool2d(enc1, kernel_size=2, stride=2)
        enc2 = self.encoder2(enc2)
        enc3 = nn.functional.max_pool2d(enc2, kernel_size=2, stride=2)
        enc3 = self.encoder3(enc3)
        enc4 = nn.functional.max_pool2d(enc3, kernel_size=2, stride=2)
        enc4 = self.encoder4(enc4)
        
        # Bottleneck
        bottleneck = nn.functional.max_pool2d(enc4, kernel_size=2, stride=2)
        bottleneck = self.bottleneck(bottleneck)
        
        # Decoder
        dec1 = nn.functional.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc4], dim=1)
        dec1 = self.decoder1(dec1)
        dec2 = nn.functional.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.decoder2(dec2)
        dec3 = nn.functional.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)
        dec4 = nn.functional.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec4 = self.decoder4(dec4)
        
        # Final output
        output = self.output(dec4)

        return output

class PyramidCNN(nn.Module):
    def __init__(self, input_channels, output_classes, h, w):
        super(PyramidCNN, self).__init__()
        self.h_ = h
        self.w_ = w
        self.ch1 = 8
        self.ch2 = 16
        self.ch3 = 32
        self.flat_dim = h//8*w//8*self.ch3

        # self.flat_dim = (self.pc*3) * h * w
        # self.flat_dim = 32*32*3

        # self.conv11 = nn.Conv2d(input_channels, self.pc, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(input_channels, self.ch1, kernel_size=3, padding=1)
        # self.conv22 = nn.Conv2d(3, self.pc, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.ch1, self.ch2, kernel_size=3, padding=1)
        # self.conv33 = nn.Conv2d(3, self.pc, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.ch2, self.ch3, kernel_size=3, padding=1)
        # self.conv44 = nn.Conv2d(3, self.pc, kernel_size=7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, output_classes)


    def forward(self, x):
        # x1 = F.relu(self.conv11(x)) 
        x = x.view(-1,6,self.h_, self.w_)
        x = self.pool(F.relu(self.conv1(x)))  # Reduce dimensions

        # x2 = F.relu(self.conv22(x))
        x = self.pool(F.relu(self.conv2(x)))  # Reduce dimensions

        # x3 = F.relu(self.conv33(x))
        x = self.pool(F.relu(self.conv3(x)))  # Reduce dimensions

        # x4 = F.relu(self.conv44(x))

        # x = torch.cat([x1,x2,x3], dim=1)
        x = x.view(-1, self.flat_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.Softmax(dim=1)(x)

        
if __name__ == "__main__":
    x = torch.rand((256,256,6))
    h, w, in_channels = x.shape
    out_classes = 9
    model = PyramidCNN(in_channels, out_classes, h, w)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels_train.csv'), 
                                         os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'))
    test_data = LogoTransformDataset(os.path.join(os.getcwd(), 'logone', 'utilities', 'labels_test.csv'),
                                     os.path.join(os.getcwd(), 'logone', 'utilities', 'transformed_256'))
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    num_epochs = 10
    # Train the model
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')

    # print('num parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))

    # out = net(x)
    # print(out.shape)