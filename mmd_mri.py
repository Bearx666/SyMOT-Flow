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

from utils.dataset import MRIDataset
from utils.vqvae_2 import VQVAE

import time



'''
    Consider to choose:
    1. 'wide_resnet50_2', # choose the feature space as 2048,8,8
    2. 'vit_base_resnet50d_224'
'''
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
        convbnrelu_2d(mid_channels, out_channels)
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

class res_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(res_conv, self).__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False))
        self.skip_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        )
        self.conv_bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
        ) 
        self.skip_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same', bias=False)
        )
        
    def forward(self, x):
        x_skip = x

        x = self.conv_bn_relu_1(x)
        x_skip = self.skip_1(x_skip)
        x = self.conv_bn_relu_2(x)
        x_skip = self.skip_2(x_skip)
        return x + x_skip
    
class Decoder(nn.Module):
    def __init__(self, input_shape) -> None:
        super(Decoder, self).__init__()

        feature_channels = input_shape[0]

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # doubleconv(feature_channels, feature_channels // 2),
            res_conv(feature_channels, feature_channels // 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # doubleconv(feature_channels // 2, feature_channels // 4),
            res_conv(feature_channels // 2, feature_channels // 4)
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # doubleconv(feature_channels // 4, feature_channels // 8),
        )
        # self.out_conv = convbnrelu_2d(feature_channels // 4, 3)
        self.out_conv = res_conv(feature_channels // 4, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        feature_x = self.decoder(x)
        out_x = self.out_conv(feature_x)
        
        return out_x
    
    
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
    def restart(self):
        self.count = 0
        self.val = 0
        
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
        

class MMDMRITrainer(object):
    def __init__(self, args=None):
        self.args = args
        
        self.n_epochs = args.n_epochs
        self.input_size = args.input_size
        self.n_flows=args.n_flows
        self.conv3x3_only=args.conv3x3_only
        self.hidden_ratio=args.hidden_ratio
        self.clamp=args.clamp

        self.lr_t = args.lr_t
        self.lr_b = args.lr_b
        self.weight_decay=args.weight_decay

        self.batch_size_tr = args.batch_size_tr
        self.batch_size_ts = args.batch_size_ts

        self.weight_ot = args.weight_ot

        self.device = torch.device('cuda', index=0)
    
    def initilize(self):
        
        self.vqvae_t1 = VQVAE().to(self.device)
        self.vqvae_t2 = VQVAE().to(self.device)

        self.vqvae_t1.load_state_dict(torch.load('vqvae_results_Task_002_MRI_T1_T2/vqvae_conv_t1_1.pth'), strict=True)
        self.vqvae_t2.load_state_dict(torch.load('vqvae_results_Task_002_MRI_T1_T2/vqvae_conv_t2_1.pth'), strict=True)

        for p_1, p_2 in zip(self.vqvae_t1.parameters(), self.vqvae_t2.parameters()):
            p_1.requires_grad = False
            p_2.requires_grad = False

        self.vqvae_t1.eval()
        self.vqvae_t2.eval()

        self.nf_flow_t  = nf_flow(
            # input_size=(256, int(self.input_size[0] // 8), int(self.input_size[1] // 8)),
            input_size = (128, 24, 24), # /8
            n_flows=self.n_flows,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
            clamp=self.clamp
        ).to(self.device)
        # self.nf_flow.load_state_dict(torch.load('mmd_results_Task_008_CT_MR/20231113-091855/params/nf_flow_ckpt_best_for_train.pth'), strict=True)

        self.nf_flow_b  = nf_flow(
            # input_size=(256, int(self.input_size[0] // 8), int(self.input_size[1] // 8)),
            input_size = (128, 48, 48), # /4
            n_flows=self.n_flows,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
            clamp=self.clamp
        ).to(self.device)

        self._ds_source_train = MRIDataset(train=1, img_type='T1')
        self._ds_target_train = MRIDataset(train=1, img_type='T2')

        self._ds_source_test = MRIDataset(train=0, img_type='T1')
        self._ds_target_test = MRIDataset(train=0, img_type='T2')


        self._ds_train = PairDateset(self._ds_source_train, self._ds_target_train)
        self._dl_train = DataLoader(self._ds_train, batch_size=self.batch_size_tr, shuffle=True, drop_last=True)

        self._ds_test = PairDateset(self._ds_source_test, self._ds_target_test)
        self._dl_test = DataLoader(self._ds_test, batch_size=self.batch_size_ts, shuffle=False, drop_last=False)

        # print(f'# of training dataset: {self._ds_source_train.__len__()}')
        # print(f'# of test dataset: {self._ds_source_test.__len__()}')

        self.optimizer_t = torch.optim.AdamW(self.nf_flow_t.parameters(), lr=self.lr_t, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.optimizer_b = torch.optim.AdamW(self.nf_flow_b.parameters(), lr=self.lr_b, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(self.nf_flow.parameters(), lr=self.lr_init, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        self.lr_schedular_t = PolyLRScheduler(self.optimizer_t, self.lr_t, self.n_epochs)
        self.lr_schedular_b = PolyLRScheduler(self.optimizer_b, self.lr_b, self.n_epochs)
        # self.lr_schedular = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        self.loss_avg_train = AverageCounter()
        self.loss_avg_test = AverageCounter()

        # self.best_test_loss = np.inf
        # self.best_train_loss = np.inf

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

        self.img_t1_dir = os.path.join(self.output_dir, 'T1')
        os.makedirs(self.img_t1_dir, exist_ok=True)
        self.img_t2_dir = os.path.join(self.output_dir, 'T2')
        os.makedirs(self.img_t2_dir, exist_ok=True)
        self.param = os.path.join(self.output_dir, 'params')
        os.makedirs(self.param, exist_ok=True)

        json_str = json.dumps(argsDict, indent=4)
        with open(args_path, 'w') as json_file:
            json_file.write(json_str)
        self.logger = Logger(str(self.output_dir) +'/train.log')

    def train_process(self):
        start_time = time.time()
        self.initilize()
        self.build_result_pth()

        for epoch in range(self.n_epochs):
            self.logger.append('-'*50)
            self.loss_of_training[epoch], img_t1_for_save_tr, img_t2_for_save_tr = self.train_one_epoch(epoch)
            self.loss_of_test[epoch], img_t1_for_save_ts, img_t2_for_save_ts = self.test_one_epoch(epoch)
            if self.args.use_lr_schedular:
                self.lr_schedular_t.step(epoch + 1)
                self.lr_schedular_b.step(epoch + 1)
            self.lr_list[epoch] = self.optimizer_b.param_groups[0]['lr']

            torch.save(self.nf_flow_t.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_ckpt_cur_t.pth'))
            torch.save(self.nf_flow_b.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_ckpt_cur_b.pth'))
            self.draw_training_and_test_loss(epoch)

            if epoch % 10 == 0:
                torchvision.utils.save_image(torch.concat([img_t1_for_save_tr, img_t1_for_save_ts]), os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=10)  
                torchvision.utils.save_image(torch.concat([img_t2_for_save_tr, img_t2_for_save_ts]), os.path.join(self.img_t2_dir, 'img_' + str(epoch) + '.png'), nrow=10) 
            if epoch % 100 == 0 or epoch == self.n_epochs - 1:
                torch.save(self.nf_flow_t.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_t_ckpt_' + str(epoch) + '.pth'))
                torch.save(self.nf_flow_b.state_dict(), os.path.join(self.output_dir, 'params', 'nf_flow_b_ckpt_' + str(epoch) + '.pth'))
            self.logger.append(f'Epoch: {epoch}, Total time cost: {time.time() - start_time} sec')

    def train_one_epoch(self, epoch):
        # self.nf_flow_t.train()
        # self.nf_flow_b.train()
        self.loss_avg_train.restart()
        for img_t1, img_t2 in self._dl_train:
        # for (img_t1, img_t2)in zip(self._dl_source_train, self._dl_target_train):
            img_t1, img_t2 = img_t1.to(self.device), img_t2.to(self.device)

            # f_t1 = self.encoder(img_t1)[-4]
            # f_t2 = self.encoder(img_t2)[-4]

            f_t1_t, f_t1_b = self.vqvae_t1.encode_1(img_t1)
            f_t2_t, f_t2_b = self.vqvae_t2.encode_1(img_t2)

            f_t2_fake, _ = self.nf_flow_t(f_t1_t)
            f_t1_fake, _ = self.nf_flow_t(f_t2_t, rev=True)

            f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1_t), flatten(f_t2_t), flatten(f_t1_fake), flatten(f_t2_fake)
            loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=10) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=10)
            penalty_ot = 0.5 * torch.mean((f_t1_ft - f_t2_fake_ft) ** 2) + 0.5 * torch.mean((f_t2_ft - f_t1_fake_ft) ** 2)
            loss_t = loss_mmd + self.weight_ot * penalty_ot

            self.optimizer_t.zero_grad()
            loss_t.backward()
            self.optimizer_t.step()

            # ------------------------------------------------------------

            f_t2_fake, _ = self.nf_flow_b(f_t1_b)
            f_t1_fake, _ = self.nf_flow_b(f_t2_b, rev=True)

            f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1_b), flatten(f_t2_b), flatten(f_t1_fake), flatten(f_t2_fake)
            loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=10) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=10)
            penalty_ot = 0.5 * torch.mean((f_t1_ft - f_t2_fake_ft) ** 2) + 0.5 * torch.mean((f_t2_ft - f_t1_fake_ft) ** 2)
            loss_b = loss_mmd + self.weight_ot * penalty_ot

            self.optimizer_b.zero_grad()
            loss_b.backward()
            self.optimizer_b.step()

            loss = 0.5 * (loss_t + loss_b)

            self.loss_avg_train.addval(loss.cpu().detach().item())

        self.logger.append(f'[Training] Epoch: {epoch}, Training Loss: {self.loss_avg_train.getmean:.6f}')


        with torch.no_grad():
            f_t1_t, f_t1_b = self.vqvae_t1.encode_1(img_t1)
            f_t2_t, f_t2_b = self.vqvae_t2.encode_1(img_t2)

            f_t2_fake, _ = self.nf_flow_t(f_t1_t)
            f_t1_fake, _ = self.nf_flow_t(f_t2_t, rev=True)
            f_t1_fake_t = f_t1_fake.contiguous()
            f_t2_fake_t = f_t2_fake.contiguous()

            f_t2_fake, _ = self.nf_flow_b(f_t1_b)
            f_t1_fake, _ = self.nf_flow_b(f_t2_b, rev=True)
            f_t1_fake_b = f_t1_fake.contiguous()
            f_t2_fake_b = f_t2_fake.contiguous()

            img_t1_ae = self.vqvae_t1.decode_1(f_t1_t.detach(), f_t1_b.detach())
            img_t1_fake = self.vqvae_t1.decode_1(f_t1_fake_t.detach(), f_t1_fake_b.detach())
            img_t2_ae = self.vqvae_t2.decode_1(f_t2_t.detach(), f_t2_b.detach())
            img_t2_fake = self.vqvae_t2.decode_1(f_t2_fake_t.detach(), f_t2_fake_b.detach())


            # f_t1 = self.ae_t1.encode(img_t1)
            # f_t2 = self.ae_t2.encode(img_t2)

            # f_t2_fake, _ = self.nf_flow(f_t1)
            # f_t1_fake, _ = self.nf_flow(f_t2, rev=True)

            # img_t1_ae = self.ae_t1.decode(f_t1)
            # img_t1_fake = self.ae_t1.decode(f_t1_fake)
            # img_t2_ae = self.ae_t2.decode(f_t2)
            # img_t2_fake = self.ae_t2.decode(f_t2_fake)
        

        # if epoch % 10 == 0:
            img_t1_for_save = torch.concat([img_t1_ae[:10].cpu(), img_t1_fake[:10].cpu()], dim=0)
                # torchvision.utils.save_image(img_t1_for_save, os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show)  
            img_t2_for_save = torch.concat([img_t2_ae[:10].cpu(), img_t2_fake[:10].cpu()], dim=0)
            # torchvision.utils.save_image(img_t2_for_save, os.path.join(self.img_t2_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show) 
        

        return self.loss_avg_train.getmean, img_t1_for_save, img_t2_for_save
    
    def test_one_epoch(self, epoch):
        with torch.no_grad():
            self.loss_avg_test.restart()
            for img_t1, img_t2 in self._dl_test:
            # for img_t1, img_t2 in zip(self._dl_source_test, self._dl_target_test):
                img_t1, img_t2 = img_t1.to(self.device), img_t2.to(self.device)

                f_t1_t, f_t1_b = self.vqvae_t1.encode_1(img_t1)
                f_t2_t, f_t2_b = self.vqvae_t2.encode_1(img_t2)

                f_t2_fake, _ = self.nf_flow_t(f_t1_t)
                f_t1_fake, _ = self.nf_flow_t(f_t2_t, rev=True)
                f_t1_fake_t = f_t1_fake.contiguous()
                f_t2_fake_t = f_t2_fake.contiguous()

                f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1_t), flatten(f_t2_t), flatten(f_t1_fake), flatten(f_t2_fake)
                loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=20) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=20)
                penalty_ot = 0.5 * torch.mean((f_t1_ft - f_t2_fake_ft) ** 2) + 0.5 * torch.mean((f_t2_ft - f_t1_fake_ft) ** 2)
                loss_t = loss_mmd + self.weight_ot * penalty_ot


                f_t2_fake, _ = self.nf_flow_b(f_t1_b)
                f_t1_fake, _ = self.nf_flow_b(f_t2_b, rev=True)
                f_t1_fake_b = f_t1_fake.contiguous()
                f_t2_fake_b = f_t2_fake.contiguous()

                f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1_b), flatten(f_t2_b), flatten(f_t1_fake), flatten(f_t2_fake)
                loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=20) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=20)
                penalty_ot = 0.5 * torch.mean((f_t1_ft - f_t2_fake_ft) ** 2) + 0.5 * torch.mean((f_t2_ft - f_t1_fake_ft) ** 2)
                loss_b = loss_mmd + self.weight_ot * penalty_ot

                loss = 0.5 * (loss_t + loss_b)

                img_t1_ae = self.vqvae_t1.decode_1(f_t1_t.detach(), f_t1_b.detach())
                img_t1_fake = self.vqvae_t1.decode_1(f_t1_fake_t.detach(), f_t1_fake_b.detach())
                img_t2_ae = self.vqvae_t2.decode_1(f_t2_t.detach(), f_t2_b.detach())
                img_t2_fake = self.vqvae_t2.decode_1(f_t2_fake_t.detach(), f_t2_fake_b.detach())
                
                self.loss_avg_test.addval(loss.cpu().detach().item())

        self.logger.append(f'[Test] Epoch: {epoch}, Test Loss: {self.loss_avg_test.getmean:.6f}')
                

        # if epoch % 10 == 0:
        img_t1_for_save = torch.concat([img_t1_ae[:10].cpu(), img_t1_fake[:10].cpu()], dim=0)
            # torchvision.utils.save_image(img_t1_for_save, os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show)  
        img_t2_for_save = torch.concat([img_t2_ae[:10].cpu(), img_t2_fake[:10].cpu()], dim=0)
            # torchvision.utils.save_image(img_t2_for_save, os.path.join(self.img_t2_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show) 
        
        return self.loss_avg_test.getmean, img_t1_for_save, img_t2_for_save
    
    def draw_training_and_test_loss(self, epoch):
        fig, ax_all= plt.subplots(2,1,figsize=(12,9))
        xvalues = np.arange(epoch + 1)

        ax = ax_all[0]
        ax.plot(xvalues, self.loss_of_training[:epoch+1], color='b', ls='-', label='loss_tr', linewidth=2)
        ax.plot(xvalues, self.loss_of_test[:epoch+1], color='r', ls='-', label='loss_ts', linewidth=2)
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
    
    def train_vqvae(self, img_type='T1'):
        assert img_type in ['T1', 'T2']
        output_dir = 'vqvae_results_' + self.args.task_name
        os.makedirs(output_dir, exist_ok=True)
        ckpt_dir = os.path.join('utils', self.args.task_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        ds = MRIDataset(img_type=img_type, train=1)
        dl = DataLoader(ds, batch_size=100, shuffle=True, drop_last=True)

        net = VQVAE(
            in_channel=1
        ).to(self.device)
        
        # net.load_state_dict(torch.load('vqvae_results_Task_002_MRI_T1_T2/vqvae_conv_' + img_type.lower() +'_0.pth'), strict=True)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_schedular = PolyLRScheduler(optimizer, 1e-3, 1001)
        loss_func = torch.nn.MSELoss()

        l_count = AverageCounter()
        for epoch in range(0, 1001):
            net.train()
            l_count.restart()
            for data in dl:
                net.zero_grad()
                data = data.to(self.device)
                dec, diff = net(data)
                
                recon_loss = loss_func(dec, data)
                latent_loss = diff.mean()

                loss = recon_loss + 1. * latent_loss

                loss.backward()
                optimizer.step()

                l_count.addval(loss.detach().cpu().item())

            if epoch % 100 == 0:
                net.eval()
                with torch.no_grad():
                    data = data[:10]
                    dec, _ = net(data)
                    img_for_save = torch.concat([data[:10], dec[:10]], dim=0)
                    torchvision.utils.save_image(img_for_save.cpu(), output_dir + '/img_' + str(epoch) + '.png', nrow=10)
                torch.save(net.state_dict(), os.path.join(output_dir, 'vqvae_gray_' + img_type.lower() + '_0.pth'))

            print(f'[Epoch: {epoch}], loss: {l_count.getmean:.6f}')
            lr_schedular.step()
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD for MRI between T1 and T2')
    parser.add_argument('--task_name', type=str, default='Task_002_MRI_T1_T2')

    parser.add_argument('--input_size', type=tuple, default=((192,192)))
    parser.add_argument('--n_flows', type=int, default=8)
    parser.add_argument('--conv3x3_only', type=bool, default=False)
    parser.add_argument('--hidden_ratio', type=float, default=1.)
    parser.add_argument('--clamp', type=float, default=2.)
    parser.add_argument('--n_epochs', type=int, default=300)

    parser.add_argument('--lr_init', type=float, default=5e-3)
    parser.add_argument('--lr_t', type=float, default=1e-3)
    parser.add_argument('--lr_b', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--batch_size_tr', type=int, default=24)
    parser.add_argument('--batch_size_ts', type=int, default=24)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--use_lr_schedular', default=False, action='store_true')

    parser.add_argument('--weight_ot', default=1e-2, type=float)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_idx)
    mmd_mri_trainer = MMDMRITrainer(args=args)

    # mmd_mri_trainer.train_vqvae('T1')
    # mmd_mri_trainer.train_vqvae('T2')
    mmd_mri_trainer.train_process()
    
 
    