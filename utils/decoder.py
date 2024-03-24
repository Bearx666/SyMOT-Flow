import timm
import torch 
import torchvision
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch.nn.functional as F


class convbnrelu_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(convbnrelu_2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        # self.bn = nn.BatchNorm2d(num_features=out_channels, affine=False, track_running_stats=False)
        # self.relu = nn.ReLU(inplace=True)
        self.bn = nn.GroupNorm(32, num_channels=out_channels) # maybe need to change the group number
        self.relu = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
def single_conv(in_dim, out_dim):
    num_groups = 32
    if in_dim <= 32:
        num_groups = 1
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding='same'),
        nn.GroupNorm(num_groups, out_dim),
        # nn.BatchNorm3d(out_dim),
    )

def double_conv(in_dim, mid_dim, out_dim):
    return nn.Sequential(
        single_conv(in_dim=in_dim, out_dim=mid_dim),
        nn.ReLU(),
        single_conv(in_dim=mid_dim, out_dim=out_dim)
    )

# start from the features [1024, w/16, h/16]
class autoencoder(nn.Module):
    def __init__(self) -> None:
        super(autoencoder, self).__init__()

        # self.encoder = timm.create_model(
        #     'wide_resnet50_2',
        #     pretrained=True,
        #     features_only=True,
        #     out_indices=[0,1,2,3],
        # )
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        # self.conv1 = double_conv(1024, 512, 512)
        self.conv2 = double_conv(256, 64, 64)
        self.conv3 = double_conv(64, 32, 32)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        self.encoder.eval()
        # _, _, features, _ = self.encoder(x)
        features = self.encoder(x)
        # f1 = F.interpolate(features, scale_factor=2, mode='nearest')
        # f1 = self.conv1(f1)
        f2 = F.interpolate(features, scale_factor=2, mode='bilinear')
        f2 = self.conv2(f2)
        f3 = F.interpolate(f2, scale_factor=2, mode='bilinear')
        f3 = self.conv3(f3)
        # f4 = F.interpolate(f3, scale_factor=2, mode='bilinear')
        img_rec = self.conv4(f3)

        return img_rec

class TrainDateset(Dataset):
    def __init__(self, petorct):
        self.dataset = glob(os.path.join('./data/PETCT', petorct + '_*.png'))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        img = Image.open(self.dataset[index]).convert('RGB')
        img = self.transforms(img)
        return img
    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    os.makedirs('./utils/ae_img', exist_ok=True)

    device = 'cuda:4'
    n_epochs = 1000
    lr_init = 1e-3
    batch_size = 128
    petorct = 'PET'

    model = autoencoder().cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-5)

    dataset = TrainDateset(petorct=petorct)
    # print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for i in range(n_epochs):

        for data in dataloader:
            img = data.cuda()
            img_rec = model(img)
            loss = criterion(img_rec, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # mse_loss = nn.MSELoss()(img_rec, img)

        print('iter: {}, loss: {}'.format(i, loss.item()))

        if i % 100 == 0:
            torchvision.utils.save_image(img_rec, './utils/ae_img/rec_' + str(i) + '.png')
            torchvision.utils.save_image(data, './utils/ae_img/ori_' + str(i) + '.png')
    torch.save(model.state_dict(), './utils/autoencoder_' + petorct + '.pth')