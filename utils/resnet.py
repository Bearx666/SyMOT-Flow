import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from utils.att_resnet import AttentionBlock


# class ResidualBlock(nn.Module):
#     def __init__(self, in_c, out_c) :
#         super(ResidualBlock, self).__init__()
        
#         self.in_norm = nn.BatchNorm2d(in_c)
#         # self.in_conv = nn.utils.weight_norm(
#         #     nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
#         # ) 
#         self.in_conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
#         self.out_norm = nn.BatchNorm2d(out_c)
#         # self.out_conv = nn.utils.weight_norm(
#         #     nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
#         # )
#         self.out_conv = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')

    
#     def forward(self, x):
#         skip = x

#         x = self.in_norm(x)
#         x = F.relu(x)
#         x = self.in_conv(x)

#         x = self.out_norm(x)
#         x = F.relu(x)
#         x = self.out_conv(x)

#         x  = x + skip

#         return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_c, out_c, n_blocks=4, mid_dim=256, attention=[2,4]) -> None:
        super(ResNet, self).__init__()
        # mid_dim = int(in_c * hidden_ratio)
        self.in_norm = nn.BatchNorm2d(in_c)
        self.in_conv = nn.Conv2d(in_c, mid_dim, kernel_size=3, padding='same')

        self.in_skip = nn.Conv2d(in_c, mid_dim, kernel_size=1, padding='same')

        self.skips = nn.ModuleList([
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1, padding='same', bias=False)
            for _ in range(n_blocks)
        ])
        self.out_conv = nn.Conv2d(mid_dim, out_c, kernel_size=1, padding='same') 

        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(BasicBlock(mid_dim, mid_dim))
            if i + 1 in attention:
                self.blocks.append(AttentionBlock(mid_dim, num_heads=4))
        self.blocks = nn.Sequential(*self.blocks)

        # self.blocks = nn.ModuleList([
        #     BasicBlock(mid_dim, mid_dim) for _ in range(n_blocks)
        # ])

        self.out_norm = nn.BatchNorm2d(mid_dim) 
    
    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x_skip = self.in_skip(x)
        x = self.in_conv(x)
        

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x


