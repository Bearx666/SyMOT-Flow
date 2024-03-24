import torch
from torch import nn
from torch.nn import functional as F
import timm

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride=4):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    

class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    

class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128, #256,
        n_res_block=2, #4,
        n_res_channel=32, #64,
        stride=4 #2
    ):
        super().__init__()

        self.encoder = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        self.decoder = Decoder(channel, in_channel, channel, n_res_block, n_res_channel, stride=stride)
    
    def forward(self, x):
        features = self.encode(x)
        out = self.decode(features)
        return out
    def encode(self, x):
        return self.encoder(x)
    def decode(self, features):
        return self.decoder(features)
    
if __name__ == '__main__':
    x = torch.ones(5, 3, 224, 168)
    net = AutoEncoder(stride=4)
    features = net.encode(x)
    print(features.shape)
    out = net.decode(features)
    print(out.shape)

    encoder = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True)
    en_f = encoder(x)
    print(en_f[-4].shape)
     
