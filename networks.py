# example network classes
# the examples here are LeNet-5 and AlexNet (CNNs).
# you can get more complex predefined and pre-trained networks (like VGG, MobileNet, ResNet, etc.) from torchvision.models


import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), # 28*28*6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 14*14*6
            
            nn.Conv2d(6, 16, kernel_size=5), # 10*10*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 5*5*16
            
            nn.Conv2d(16, 120, kernel_size=5), # 1*1*120
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 84), # 84
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes), # 10
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # input size: 224*224*3
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 55*55*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27*27*64

            nn.Conv2d(64, 192, kernel_size=5, padding=2), # 27*27*192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 13*13*192

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 13*13*384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 13*13*256
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 13*13*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 6*6*256
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), # 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), # 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), # 1000
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.classifier(x)
        return x