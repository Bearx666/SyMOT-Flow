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

class LoadLabelMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, label_pth, input_size, img_type='mnist'):
        self.files, _ = load_data(os.path.join(root, img_type.upper(), 'raw'))
        label_list = np.load(label_pth)
        self.files = [self.files[i][label_list] for i in range(len(self.files))]

        self.transforms = transforms.Compose([
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __getitem__(self, index):
        img, label = self.files[0][index], self.files[1][index]
        img_rgb = Image.fromarray(img).convert('RGB')
        img_ts = self.transforms(img_rgb)

        return img_ts, label

    def __len__(self):
        return len(self.files[0])
        

def make_ssl_dataloaders(
        root,
        input_size,
        label_path,
        batch_size_labeled,
        batch_size_unlabeled,
        transform_tr,
        transform_ts,
        dataset='mnist',
        iter_num = 500
    ):
    assert dataset in ['mnist', 'fmnist']

    if dataset == 'mnist':
        ds_tr = MNISTDataset(root=root, input_size=input_size, train=True, img_type='mnist')
        ds_ts = MNISTDataset(root=root, input_size=input_size, train=False, img_type='mnist')
    elif dataset == 'fmnist':
        ds_tr = MNISTDataset(root=root, input_size=input_size, train=True, img_type='fmnist')
        ds_ts = MNISTDataset(root=root, input_size=input_size, train=False, img_type='fmnist')

    label_npy = np.load(label_path)
    unlabel_npy = np.array(list(filter(lambda x: x not in label_npy, np.arange(ds_tr.__len__()))))

    print(f'# of labels: {len(label_npy)}, # of unlabels: {len(unlabel_npy)}')

    idx = np.arange(ds_tr.__len__())
    unlabel_idx = idx[:len(unlabel_npy)]
    label_idx = idx[len(unlabel_npy):]

    ds_tr.files[0] = np.vstack([
        ds_tr.files[0][unlabel_npy],
        ds_tr.files[0][label_npy]
    ])
    ds_tr.files[1] = np.hstack([
        ds_tr.files[1][unlabel_npy],
        ds_tr.files[1][label_npy]
    ])

    ds_tr.files[1][unlabel_idx] = -1

    num_train = len(ds_tr.files[0])
    
    assert num_train == len(label_idx) + len(unlabel_idx)

    print(f'Data Name: {dataset}, Labeled data: {len(label_idx)}, Unlabeled data: {len(unlabel_idx)}, Num batch: {iter_num}')

    batch_sampler = LabeledUnlabeledBatchSampler(label_idx, unlabel_idx, batch_size_labeled, batch_size_unlabeled, iter_num)

    dl_tr = DataLoader(
        dataset=ds_tr,
        batch_sampler=batch_sampler,
    )

    dl_ts = DataLoader(
        dataset=ds_ts,
        batch_size=200,
        shuffle=False,
        drop_last=False
    )

    return dl_tr, dl_ts



class LabeledUnlabeledBatchSampler(Sampler):
    """Minibatch index sampler for labeled and unlabeled indices. 

    An epoch is one pass through the labeled indices.
    """
    def __init__(
            self, 
            labeled_idx, 
            unlabeled_idx, 
            labeled_batch_size, 
            unlabeled_batch_size,
            iter_num):

        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_batch_size = unlabeled_batch_size
        self.labeled_batch_size = labeled_batch_size
        self.iter_num = iter_num

        assert len(self.labeled_idx) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_idx) >= self.unlabeled_batch_size > 0


    @property
    def num_labeled(self):
        return len(self.labeled_idx)

    def __iter__(self):
        # labeled_iter = iterate_func(self.labeled_idx, self.iter_num)
        # unlabeled_iter = iterate_func(self.unlabeled_idx, self.iter_num)
        labeled_iter = iterate_eternally(self.labeled_idx)
        unlabeled_iter = iterate_once(self.unlabeled_idx)
        return (
            labeled_batch + unlabeled_batch
            for (labeled_batch, unlabeled_batch)
            in  zip(batch_iterator(labeled_iter, self.labeled_batch_size),
                    batch_iterator(unlabeled_iter, self.unlabeled_batch_size))
        )

    def __len__(self):
        return len(self.labeled_idx) // self.labeled_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def iterate_func(indices, num):
    def infinite_shuffles(num):
        i = 0
        while i < num:
            yield np.random.permutation(indices)
            i += 1
    return itertools.chain.from_iterable(infinite_shuffles(num))


def batch_iterator(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)



class LoadLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, labeled_pth, img_type='fmnist', input_size=(28,28)):
        self.files, _ = load_data(os.path.join(root, img_type.upper(), 'raw'))
        label_list = np.load(labeled_pth)
        self.files = [self.files[i][label_list] for i in range(len(self.files))]

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
        elif train == -1:
            self.files = glob(os.path.join(root, 'Train', img_type, '*.png')) + glob(os.path.join(root, 'Test', img_type, '*.png'))
            print(f'Whole dataset, length is {len(self.files)}')

        self.transforms = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # if img_type == 't1':
        #     self.files = glob(os.path.join('/home/xiongz/programs/MMD/data/Task_001_MRI_T1_T2/T1', '*.png'))
        # elif img_type == 't2':
        #     self.files = glob(os.path.join('/home/xiongz/programs/MMD/data/Task_001_MRI_T1_T2/T2', '*.png'))
        
    def __getitem__(self, index):
        img_file = self.files[index]

        # ------  直方图均衡
        # img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        # img_normalized = cv2.equalizeHist(img)
        # img_rgb = Image.fromarray(img_normalized).convert('RGB')
        #-------    
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
        elif train == -1:
            self.files = glob(os.path.join(root, 'train_' + img_type.lower(), '*.png')) + glob(os.path.join(root, 'test_' + img_type.lower(), '*.png'))
            print(f'Whole dataset, length is {len(self.files)}')
        
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


# class CTMRDataset(torch.utils.data.Dataset):
#     def __init__(self, root='/home/xiongz/programs/MMD/data/Task_008_CT_MR', img_type='CT', train=1):
#         assert img_type in ['CT', 'MR']
#         if train == 1:
#             self.img_type = 'slc_ct_gray' if img_type == 'CT' else 'slc_mr_gray'
#             self.files = glob(os.path.join(root, self.img_type, '*.png'))
#             self.transforms = transforms.Compose([
#             # transforms.RandomResizedCrop(size=(224, 168), scale=(0.5, 1.0)),
#             transforms.ToTensor(),
#         ])
#             print('This is training dataset')
#         elif train == 0:
#             self.img_type = 'test_ct' if img_type == 'CT' else 'test_mr'
#             self.files = glob(os.path.join(root, self.img_type, '*.png'))
#             self.transforms = transforms.Compose([
#             # transforms.RandomResizedCrop(size=(224, 168), scale=(0.5, 1.0)),
#             transforms.ToTensor(),
#         ])
#             print('This is test dataset')
#         elif train == -1:
#             self.files = glob(os.path.join(root, 'slc_' + img_type.lower() + '_gray', '*.png')) + glob(os.path.join(root, 'test_' + img_type.lower(), '*.png'))
#             print('This is the whole dataset')
#             self.transforms = transforms.Compose([
#                 transforms.RandomResizedCrop(size=(224, 168), scale=(0.5, 1.0)),
#                 transforms.ToTensor(),
#                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#                 # transforms.Normalize(0.5, 0.5)
#             ])
    
#     def __getitem__(self, index):
#         file = self.files[index]
#         img = Image.open(file).convert('L')
#         img_ts = self.transforms(img)

#         return img_ts

#     def __len__(self):
#         return len(self.files)
    
class PelvisDataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/xiongz/programs/MMD/data/Task_009_pelvis', img_type='CT', train=1):
        assert img_type in ['CT', 'MR']
        assert img_type in ['CT', 'MR']
        if train == 1:
            self.files = glob(os.path.join(root, 'train_' + img_type.lower(), '*.png'))
            print(f'Training dataset, length is {len(self.files)}')
        elif train == 0:
            self.files = glob(os.path.join(root, 'test_' + img_type.lower(), '*.png'))
            print(f'Test dataset, length is {len(self.files)}')
        elif train == -1:
            self.files = glob(os.path.join(root, 'train_' + img_type.lower(), '*.png')) + glob(os.path.join(root, 'test_' + img_type.lower(), '*.png'))
            print(f'Whole dataset, length is {len(self.files)}')
        
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
        

