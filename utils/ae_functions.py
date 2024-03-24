import torch
import torch.nn as nn

def doubleconv(in_channels, out_channels, hidden_ratio=1.):
    mid_channels = int(hidden_ratio * in_channels)
    return nn.Sequential(
        convbnrelu_2d(in_channels, mid_channels),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(),
        convbnrelu_2d(mid_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class convbnrelu_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(convbnrelu_2d, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 
        
    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape) -> None:
        super(Decoder, self).__init__()

        feature_channels = input_shape[0]

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            doubleconv(feature_channels, feature_channels // 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            doubleconv(feature_channels // 2, feature_channels // 4),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # doubleconv(feature_channels // 4, feature_channels // 8),
        )
        self.out_conv = convbnrelu_2d(feature_channels // 4, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        feature_x = self.decoder(x)
        out_x = self.out_conv(feature_x)
        
        return out_x