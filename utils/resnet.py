import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c) :
        super(ResidualBlock, self).__init__()
        
        self.in_norm = nn.BatchNorm2d(in_c)
        # self.in_conv = nn.utils.weight_norm(
        #     nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        # ) 
        self.in_conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        self.out_norm = nn.BatchNorm2d(out_c)
        # self.out_conv = nn.utils.weight_norm(
        #     nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
        # )
        self.out_conv = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')

    
    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x  = x + skip

        return x

class ResNet(nn.Module):
    def __init__(self, in_c, out_c, n_blocks=8, hidden_ratio=1.0, use_weight_norm=True) -> None:
        super(ResNet, self).__init__()
        mid_dim = int(in_c * hidden_ratio)
        self.in_norm = nn.BatchNorm2d(in_c)
        if use_weight_norm:
            self.in_conv = nn.utils.weight_norm(
                nn.Conv2d(in_c, mid_dim, kernel_size=3, padding='same')
            ) 
            self.in_skip = nn.utils.weight_norm(
                nn.Conv2d(in_c, mid_dim, kernel_size=1, padding='same')
            ) 
            self.skips = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(mid_dim, mid_dim, kernel_size=1, padding='same'))
            for _ in range(n_blocks)
            ])
            self.out_conv = nn.utils.weight_norm(
            nn.Conv2d(mid_dim, out_c, kernel_size=1, padding='same')
            ) 
        else:
            self.in_conv = nn.Conv2d(in_c, mid_dim, kernel_size=3, padding='same')

            self.in_skip = nn.Conv2d(in_c, mid_dim, kernel_size=1, padding='same')

            self.skips = nn.ModuleList([
                nn.Conv2d(mid_dim, mid_dim, kernel_size=1, padding='same', bias=False)
                for _ in range(n_blocks)
            ])
            self.out_conv = nn.Conv2d(mid_dim, out_c, kernel_size=1, padding='same') 

        self.blocks = nn.ModuleList([
            ResidualBlock(mid_dim, mid_dim) for _ in range(n_blocks)
        ])

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


