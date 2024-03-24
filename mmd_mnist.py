import timm
import torch
import os
import cv2
from glob import glob

import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np

import FrEIA.framework as Ff 
import FrEIA.modules as Fm

import argparse
import pathlib
import json
import datetime
import traceback

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from torch.optim.lr_scheduler import _LRScheduler
from utils.resnet import ResNet
import gzip



'''
    Consider to choose:
    1. 'wide_resnet50_2', # choose the feature space as 2048,8,8
    2. 'vit_base_resnet50d_224'
'''

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
    return (x_train, y_train), (x_test, y_test)

def flatten(img):
    img_flatten = img.view(img.size(0), -1)
    return img_flatten

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

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

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

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath,resume=False): 
        self.file = None
        self.resume = resume
        if os.path.isfile(fpath):
            if resume:
                self.file = open(fpath, 'a') 
            else:
                self.file = open(fpath, 'w')
        else:
            self.file = open(fpath, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                # print(target_str)
                self.file.write(target_str + '\n')
                self.file.flush()
        else:
            # print(target_str)
            self.file.write(target_str + '\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

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

        # if img_type == 't1':
        #     self.files = glob(os.path.join('/home/xiongz/programs/MMD/data/Task_001_MRI_T1_T2/T1', '*.png'))
        # elif img_type == 't2':
        #     self.files = glob(os.path.join('/home/xiongz/programs/MMD/data/Task_001_MRI_T1_T2/T2', '*.png'))
        
    def __getitem__(self, index):
        # img, label = self.files[0][index], self.files[1][index]
        img = self.files[0][index]

        img_rgb = Image.fromarray(img).convert('RGB')
        img_ts = self.transforms(img_rgb)

        return img_ts #, label
        
    def __len__(self):
        return len(self.files[0])
    
class AverageCounter(object):
    def __init__(self) -> None:
        self.count = 0
        self.val = 0
    def addval(self, val):
        self.val += val
        self.count += 1
    @property
    def getmean(self):
        if self.count > 0:
            return self.val / self.count
        else:
            raise ZeroDivisionError('Count is 0 !!')
        
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                    kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_dim = int(hidden_ratio * in_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
        )
    return subnet_conv

def nf_flow(
        input_size,
        n_flows=8,
        conv3x3_only=False,
        hidden_ratio=1.0,
        clamp=2.0,
    ):

    nodes = Ff.SequenceINN(*input_size)
    for i in range(n_flows):
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=ResNet,
            affine_clamping=clamp,
            permute_soft=False,
        )
    # nodes = Ff.SequenceINN(*input_size)

    # for i in range(n_flows):
    #     if i % 2 == 1 and not conv3x3_only:
    #         kernel_size = 1
    #     else:
    #         kernel_size = 3

    #     nodes.append(
    #         Fm.AllInOneBlock,
    #         subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
    #         affine_clamping=clamp,
    #         permute_soft=False,
    #     )
    return nodes

class PairDateset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.x1 = source_dataset
        self.x2 = target_dataset
    def __getitem__(self, index):
        x1_sample = self.x1.__getitem__(index)
        x2_sample = self.x2.__getitem__(index)
        # x1_sample = self.x1[index]/255
        # x2_sample = self.x2[index]/255
        return x1_sample, x2_sample
    def __len__(self):
        return self.x1.__len__()
        

class MNISTTrainer(object):
    def __init__(self, args=None):
        self.args = args
        
        self.root = args.root
        self.n_epochs = args.n_epochs
        
        self.input_size = args.input_size
        self.n_flows=args.n_flows
        self.conv3x3_only=args.conv3x3_only
        self.hidden_ratio=args.hidden_ratio
        self.clamp=args.clamp

        self.lr_init = args.lr_init
        self.weight_decay=args.weight_decay

        self.batch_size_tr = args.batch_size_tr
        self.batch_size_ts = args.batch_size_ts

        self.device = torch.device('cuda', index=0)
    
    def initilize(self):
        # this part should be put into the initial function
        self.encoder = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True).to(self.device)
        self.decoder_t1 = Decoder(input_shape=(256, 56, 56)).to(self.device)
        self.decoder_t2 = Decoder(input_shape=(256, 56, 56)).to(self.device)

        # for p_e in self.encoder.parameters():
        #     p_e.requires_grad = False
        for p_e, p_d_t1, p_d_t2 in zip(self.encoder.parameters(), self.decoder_t1.parameters(), self.decoder_t2.parameters()):
            p_e.requires_grad = False
            p_d_t1.requires_grad = False
            p_d_t2.requires_grad = False
        
        pretrained_decoder_t1_param = np.load('./utils/Task_MNIST_FMNIST/decoder_mnist_1.npy', allow_pickle=True).item()
        pretrained_decoder_t2_param = np.load('./utils/Task_MNIST_FMNIST/decoder_fmnist_1.npy', allow_pickle=True).item()
        self.decoder_t1.load_state_dict(pretrained_decoder_t1_param['decoder_param'], strict=True)
        self.decoder_t2.load_state_dict(pretrained_decoder_t2_param['decoder_param'], strict=True)

        self.nf_flow = nf_flow(
            input_size=(256, 48, 48),
            n_flows=self.n_flows,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
            clamp=self.clamp
        ).to(self.device)
        print(f'# of training parameters: {sum([p.numel() for p in self.nf_flow.parameters() if p.requires_grad == True])}')

        self._ds_source_train = MNISTDataset(root=self.root, input_size=self.input_size, train=True, img_type='mnist')
        self._ds_target_train = MNISTDataset(root=self.root, input_size=self.input_size, train=True, img_type='fmnist')

        self._ds_source_test = MNISTDataset(root=self.root, input_size=self.input_size, train=False, img_type='mnist')
        self._ds_target_test = MNISTDataset(root=self.root, input_size=self.input_size, train=False, img_type='fmnist')

        self._dl_source_train = DataLoader(self._ds_source_train, batch_size=self.batch_size_tr, shuffle=True, drop_last=True)
        self._dl_target_train = DataLoader(self._ds_target_train, batch_size=self.batch_size_tr, shuffle=True, drop_last=True)

        self._dl_source_test = DataLoader(self._ds_source_test, batch_size=self.batch_size_ts, shuffle=True, drop_last=False)
        self._dl_target_test = DataLoader(self._ds_target_test, batch_size=self.batch_size_ts, shuffle=True, drop_last=False)

        # self._ds_train = PairDateset(self._ds_source_train, self._ds_target_train)
        # self._dl_train = DataLoader(self._ds_train, batch_size=self.batch_size_tr, shuffle=True, drop_last=True)

        # self._ds_test = PairDateset(self._ds_source_test, self._ds_target_test)
        # self._dl_test = DataLoader(self._ds_test, batch_size=self.batch_size_ts, shuffle=False, drop_last=False)

        print(f'# of training dataset: {self._ds_source_train.__len__()}')
        print(f'# of test dataset: {self._ds_source_test.__len__()}')

        self.optimizer = torch.optim.AdamW(self.nf_flow.parameters(), lr=self.lr_init, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(self.nf_flow.parameters(), lr=self.lr_init, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        self.lr_schedular = PolyLRScheduler(self.optimizer, self.lr_init, self.n_epochs)
        # self.lr_schedular = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        self.loss_avg_train = AverageCounter()
        self.loss_avg_test = AverageCounter()

        self.best_test_loss = np.inf
        self.best_train_loss = np.inf

        self.loss_of_training = np.zeros(self.n_epochs, )
        self.loss_of_test = np.zeros(self.n_epochs, )
        self.lr_list = np.zeros(self.n_epochs, )

    def build_result_pth(self):
        args = self.args
        output_dir = './mmd_results_' + str(args.task_name)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = (
            pathlib.Path(output_dir).resolve().joinpath(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        argsDict = args.__dict__
        args_path = str(self.output_dir) + '/args.json'

        self.img_t1_dir = os.path.join(self.output_dir, 'MNIST')
        os.makedirs(self.img_t1_dir, exist_ok=True)
        self.img_t2_dir = os.path.join(self.output_dir, 'FMNIST')
        os.makedirs(self.img_t2_dir, exist_ok=True)
        self.param = os.path.join(self.output_dir, 'params')
        os.makedirs(self.param, exist_ok=True)

        json_str = json.dumps(argsDict, indent=4)
        with open(args_path, 'w') as json_file:
            json_file.write(json_str)
        self.logger = Logger(str(self.output_dir) +'/train.log')

    def train_process(self):
        self.initilize()
        self.build_result_pth()

        for epoch in range(self.n_epochs):
            self.logger.append('-'*50)
            self.loss_of_training[epoch], img_t1_for_save_tr, img_t2_for_save_tr = self.train_one_epoch(epoch)
            self.loss_of_test[epoch], img_t1_for_save_ts, img_t2_for_save_ts = self.test_one_epoch(epoch)
            if self.args.use_lr_schedular:
                self.lr_schedular.step(epoch + 1)
                # self.lr_schedular.step()
            self.lr_list[epoch] = self.optimizer.param_groups[0]['lr']
            torch.save(self.nf_flow.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_ckpt_current.pth'))
            self.draw_training_and_test_loss(epoch)

            if epoch % 10 == 0:
                img_t1_for_save_all = torch.concat([img_t1_for_save_tr, img_t1_for_save_ts], dim=0)
                img_t2_for_save_all = torch.concat([img_t2_for_save_tr, img_t2_for_save_ts], dim=0)
                torchvision.utils.save_image(img_t1_for_save_all, os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=5)  
                torchvision.utils.save_image(img_t2_for_save_all, os.path.join(self.img_t2_dir, 'img_' + str(epoch) + '.png'), nrow=5) 
            if epoch % 100 == 0:
                torch.save(self.nf_flow.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_ckpt_' + str(epoch) + '.pth'))

    def train_one_epoch(self, epoch):
        self.nf_flow.train()
        # for img_t1, img_t2 in self._dl_train:
        for (img_t1, img_t2)in zip(self._dl_source_train, self._dl_target_train):
            img_t1, img_t2 = img_t1.to(self.device), img_t2.to(self.device)
            f_t1 = self.encoder(img_t1)[-4]
            f_t2 = self.encoder(img_t2)[-4]

            f_t2_fake, _ = self.nf_flow(f_t1)
            f_t1_fake, _ = self.nf_flow(f_t2, rev=True)

            f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1), flatten(f_t2), flatten(f_t1_fake), flatten(f_t2_fake)
            loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=10) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=10)

            # loss = loss_mmd + loss_recon
             
            self.optimizer.zero_grad()
            loss_mmd.backward()
            self.optimizer.step()

            self.loss_avg_train.addval(loss_mmd.cpu().detach().item())

        self.logger.append(f'[Training] Epoch: {epoch}, Training Loss: {self.loss_avg_train.getmean:.6f}')

        if self.best_train_loss > self.loss_avg_train.getmean:
            self.best_train_loss = self.loss_avg_train.getmean
            torch.save(self.nf_flow.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_ckpt_best_for_train.pth'))
            self.logger.append(f'Best [training] parameters are refreshed on [Epoch:{epoch}]')

        img_t1_ae = self.decoder_t1(f_t1.detach())
        img_t1_fake = self.decoder_t1(f_t1_fake.detach())
        img_t2_ae = self.decoder_t2(f_t2.detach())
        img_t2_fake = self.decoder_t2(f_t2_fake.detach())

        # if epoch % 10 == 0:
        img_t1_for_save = torch.concat([img_t1_ae[:5].cpu(), img_t1_fake[:5].cpu()], dim=0)
            # torchvision.utils.save_image(img_t1_for_save, os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show)  
        img_t2_for_save = torch.concat([img_t2_ae[:5].cpu(), img_t2_fake[:5].cpu()], dim=0)
            # torchvision.utils.save_image(img_t2_for_save, os.path.join(self.img_t2_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show) 
        

        return self.loss_avg_train.getmean, img_t1_for_save, img_t2_for_save
    
    def test_one_epoch(self, epoch):
        self.nf_flow.eval()
        # for img_t1, img_t2 in self._dl_test:
        for img_t1, img_t2 in zip(self._dl_source_test, self._dl_target_test):
            img_t1, img_t2 = img_t1.to(self.device), img_t2.to(self.device)
            f_t1 = self.encoder(img_t1)[-4]
            f_t2 = self.encoder(img_t2)[-4]
            f_t2_fake, _ = self.nf_flow(f_t1)
            f_t1_fake, _ = self.nf_flow(f_t2, rev=True)

            f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1), flatten(f_t2), flatten(f_t1_fake), flatten(f_t2_fake)
            loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=10) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=10)

            self.loss_avg_test.addval(loss_mmd.cpu().detach().item())

        self.logger.append(f'[Testing] Epoch: {epoch}, Test Loss: {self.loss_avg_test.getmean:.6f}')
        if self.best_test_loss > self.loss_avg_test.getmean:
            self.best_test_loss = self.loss_avg_test.getmean
            torch.save(self.nf_flow.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_ckpt_best_for_test.pth'))
            self.logger.append(f'Best [Test] parameters are refreshed on [Epoch:{epoch}]')

        img_t1_ae = self.decoder_t1(f_t1.detach())
        img_t1_fake = self.decoder_t1(f_t1_fake.detach())
        img_t2_ae = self.decoder_t2(f_t2.detach())
        img_t2_fake = self.decoder_t2(f_t2_fake.detach())

        # if epoch % 10 == 0:
        img_t1_for_save = torch.concat([img_t1_ae[:5].cpu(), img_t1_fake[:5].cpu()], dim=0)
            # torchvision.utils.save_image(img_t1_for_save, os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show)  
        img_t2_for_save = torch.concat([img_t2_ae[:5].cpu(), img_t2_fake[:5].cpu()], dim=0)
            # torchvision.utils.save_image(img_t2_for_save, os.path.join(self.img_t2_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show) 
        
        return self.loss_avg_test.getmean, img_t1_for_save, img_t2_for_save

    def draw_training_and_test_loss(self, epoch):
        fig, ax_all= plt.subplots(2,1,figsize=(12,9))
        xvalues = np.arange(epoch + 1)

        ax = ax_all[0]
        ax.plot(xvalues, self.loss_of_training[:epoch+1], color='b', ls='-', label='loss_tr', linewidth=2)
        ax.plot(xvalues, self.loss_of_test[:epoch+1], color='red', ls='-', label='loss_ts', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc=(0, 1))

        ax = ax_all[1]
        ax.plot(xvalues, self.lr_list[:epoch+1], ls='-', label='lr', linewidth=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend(loc=(0, 1))

        fig.savefig(os.path.join(self.output_dir, 'process.png'))
        plt.close()
    
    def pretain_decoder(self, img_type='mnist'):
        os.makedirs('utils/Task_MNIST_FMNIST', exist_ok=True)
        os.makedirs('test_for_mnist', exist_ok=True)

        encoder = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True)
        for param in encoder.parameters():
            param.requires_grad = False
        decoder = Decoder(input_shape=(256, 7, 7))

        ds_train = MNISTDataset(root='data', input_size=(28,28), img_type=img_type)
        print(f'The length of Training Dataset: {ds_train.__len__()}')
        dl_train = DataLoader(ds_train, batch_size=200, shuffle=True, drop_last=True)

        device = torch.device('cuda', index=0)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=5e-4)
        l_func = torch.nn.MSELoss()

        l_count = AverageCounter()
        for i in range(1, 101):
            for data in dl_train:
                # print(data.shape)
                data = data.to(device)
                features = encoder(data)[-4]
                output = decoder(features)
                # print(torch.sum(torch.abs(output), dim=(0,1,2,3)))
                l = l_func(data, output) #+ 0.5 * torch.mean(torch.abs(output), dim=(0,1,2,3))
                
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                l_count.addval(l.detach().cpu().item())
            if i % 5 == 0:
                img_for_save = torch.concat([data[:5], output[:5]], dim=0)
                torchvision.utils.save_image(img_for_save.detach().cpu(), 'test_for_mnist/img_' + str(i) + '.png', nrow=5)
                print(f'[Epoch: {i}], loss: {l_count.getmean:.6f}')
        
        d = dict()
        d['data_type'] = img_type
        d['feaure_size'] = (256, 7, 7)
        d['encoder_type'] = 'wide_resnet50_2'
        d['decoder_param'] = decoder.state_dict()
        np.save('utils/Task_MNIST_FMNIST/decoder_' + img_type.lower() + '_1.npy', d)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD between MNIST and FMNIST')
    parser.add_argument('--task_name', type=str, default='Task_004_MNIST_FMNIST')
    parser.add_argument('--root', type=str, default='data')

    parser.add_argument('--input_size', type=tuple, default=((28,28)))
    parser.add_argument('--n_flows', type=int, default=8)
    parser.add_argument('--conv3x3_only', type=bool, default=False)
    parser.add_argument('--hidden_ratio', type=float, default=1.)
    parser.add_argument('--clamp', type=float, default=2.)
    parser.add_argument('--n_epochs', type=int, default=300)

    parser.add_argument('--lr_init', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--batch_size_tr', type=int, default=32)
    parser.add_argument('--batch_size_ts', type=int, default=24)
    # parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--use_lr_schedular', default=False, action='store_true')

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_idx)
    mmd_mri_trainer = MNISTTrainer(args=args)

    mmd_mri_trainer.train_process()
    # mmd_mri_trainer.pretain_decoder('mnist')
    # mmd_mri_trainer.pretain_decoder('fmnist')
    
    






























