import gzip
import os
import numpy as np
import torch
import itertools
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from PIL import Image
from collections import defaultdict
import torchvision
from torch.utils.data import DataLoader
from glob import glob

def load_data(data_folder):
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
    with gzip.open(paths[0], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16
        ).reshape(-1, 28, 28)
    with gzip.open(paths[1], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[2], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16
        ).reshape(-1, 28, 28)
    with gzip.open(paths[3], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    return [x_train, y_train], [x_test, y_test]

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, input_size, train=True, img_type='mnist') -> None:
        assert img_type in ['mnist','fmnist'], 'Image type should be mnist or fmnist'
        self.train = train
        self.img_type = img_type

        if self.train:
            self.files, _ = load_data(os.path.join(root, img_type.upper(), 'raw'))
        else:
            _, self.files = load_data(os.path.join(root, img_type.upper(), 'raw'))

        self.transforms = transforms.Compose([
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        
    def __getitem__(self, index):
        img, label = self.files[0][index], self.files[1][index]
        # img = self.files[0][index]

        img_rgb = Image.fromarray(img).convert('RGB')
        img_ts = self.transforms(img_rgb)

        return img_ts, label
        
    def __len__(self):
        return len(self.files[0])

    
class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/xiongz/programs/MMD/data/Task_002_MRI_T1_T2', train=True, img_type='T2') -> None:
        assert img_type in ['T1', 'T2'], 'Image type should be T1 or T2'
        assert train in [-1, 0, 1], 'Train=1 means for training, Train=0 means for testing, Train=-1 means the whole dataset'
        self.train = train

        if train == 1:
            self.files = glob(os.path.join(root, 'Train', img_type, '*.png'))
            print(f'Training dataset, length is {len(self.files)}')
        elif train == 0:
            self.files = glob(os.path.join(root, 'Test', img_type, '*.png'))
            print(f'Test dataset, length is {len(self.files)}')
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

      
    def __getitem__(self, index):
        img_file = self.files[index]
  
        img_rgb = Image.open(img_file).convert('RGB')
        img_ts = self.transforms(img_rgb)
        return img_ts
        
    def __len__(self):
        return len(self.files)
    
class CTMRDataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/xiongz/programs/MMD/data/Task_008_CT_MR', img_type='CT', train=1):
        assert img_type in ['CT', 'MR']
        if train == 1:
            self.files = glob(os.path.join(root, 'train_' + img_type.lower(), '*.png'))
            print(f'Training dataset, length is {len(self.files)}')
        elif train == 0:
            self.files = glob(os.path.join(root, 'test_' + img_type.lower(), '*.png'))
            print(f'Test dataset, length is {len(self.files)}')
       
        self.transforms = transforms.Compose([
#             # transforms.RandomResizedCrop(size=(224, 168), scale=(0.5, 1.0)),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file).convert('RGB')
        img_ts = self.transforms(img)

        return img_ts

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = CTMRDataset(img_type='CT')
    dl = DataLoader(ds, batch_size=5)
    for img in dl:
        print(img.shape)
        break
    # ds = MNISTDataset(root='data', input_size=(28, 28), train=True, img_type='mnist')
    # print(ds.)
    # n_labels = 100
    # img_type = 'fmnist'
    # ds, _ = load_data(os.path.join('./data', img_type.upper(), 'raw'))
    # # x_ds, y_ds = ds[0], ds[1]
    # # idx_list = []
    
    # # label_dict = defaultdict(int)
    # # for i, y in enumerate(y_ds):
    # #     if label_dict[y] < int(n_labels // 10):
    # #         idx_list.append(i)
    # #         label_dict[y] += 1
    # # np.save('./data/Task_007_MATCH/' + img_type + '/' + str(n_labels) + '.npy', np.array(idx_list))
    # idx_list = np.load('./data/Task_007_MATCH/' + img_type + '/' + str(n_labels) + '.npy')
    # print(idx_list)
    # transforms = transforms.Compose([
    #         transforms.Resize((28, 28)),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ])
    # # ds = torchvision.datasets.FashionMNIST(root='data', train=True, transform=transforms, download=True)
    # # print(ds.data.shape)
    # # print(ds.targets.shape)
    # dl_tr, dl_ts = make_ssl_dataloaders(
    #     root='data',
    #     input_size=(28, 28),
    #     label_path='data/Task_007_MATCH/mnist/10.npy',
    #     batch_size_labeled=5,
    #     batch_size_unlabeled=20,
    #     transform_tr=transforms,
    #     transform_ts=transforms,
    #     dataset='mnist',
    #     iter_num=500
    # )

    # for i, (img, label) in enumerate(dl_tr):
    #     print(label)
    #     print(img.shape)
    #     torchvision.utils.save_image(img, 'test.png')
    #     torchvision.utils.save_image(img[label == -1], 'test1.png')
    #     break
        

