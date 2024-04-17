import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
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