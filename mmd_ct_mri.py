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
import random
import time

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from torch.optim.lr_scheduler import _LRScheduler
from utils.resnet import ResNet
from utils.dataset import CTMRDataset
from utils.vqvae_2 import VQVAE
from utils.ema import EMA



def flatten(img):
    img_flatten = img.view(img.size(0), -1)
    return img_flatten


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

def files_sort(files):
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][4:]))
    return files
        
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

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_pths):
        super().__init__()
        self.files = img_pths
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        self.transforms = transforms.Compose([
            # transforms.RandomResizedCrop(size=(224, 168), scale=(0.5, 1.0)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        if len(file) == 2:
            img0 = Image.open(file[0]).convert('RGB')
            img_ts0 = self.transforms(img0)

            img1 = Image.open(file[1]).convert('RGB')
            img_ts1 = self.transforms(img1)

            return img_ts0, img_ts1

        else:
            img = Image.open(file).convert('RGB')
            img_ts = self.transforms(img)
            return img_ts

def load_data(dataloader):
    while True:
        yield from dataloader  

class Trainer(object):
    def __init__(self, args=None):
        self.args = args
        
        self.n_epochs = args.n_epochs
        self.data_root = args.data_root
        
        self.input_size = args.input_size
        self.n_flows=args.n_flows
        self.conv3x3_only=args.conv3x3_only
        self.hidden_ratio=args.hidden_ratio
        self.clamp=args.clamp

        self.img_type_1 = args.img_type_1
        self.img_type_2 = args.img_type_2

        self.lr_t = args.lr_t
        self.lr_b = args.lr_b
        self.weight_decay=args.weight_decay

        self.batch_size_tr = args.batch_size_tr
        self.batch_size_ts = args.batch_size_ts
        self.n_batch = args.n_batch
        self.unbalanced_num = args.unbalanced_num

        self.weight_ot = args.weight_ot
        self.ckpt = args.ckpt
        self.ckpt_epoch = args.ckpt_epoch

        self.device = torch.device('cuda', index=0)
    
    def initilize(self):

        self.vqvae_t1 = VQVAE().to(self.device)
        self.vqvae_t2 = VQVAE().to(self.device)

       
        self.vqvae_t1.load_state_dict(torch.load('utils/Task_003_CT_MRI_Brain/vqvae_results_ct_1.pth'), strict=True)
        self.vqvae_t2.load_state_dict(torch.load('utils/Task_003_CT_MRI_Brain/vqvae_results_mri_1.pth'), strict=True)

        print('Successfully Loaded VQ-VAE-2!')

        for p_1, p_2 in zip(self.vqvae_t1.parameters(), self.vqvae_t2.parameters()):
            p_1.requires_grad = False
            p_2.requires_grad = False

        self.vqvae_t1.eval()
        self.vqvae_t2.eval()

        self.nf_flow_t  = nf_flow(
            input_size = (128, 28, 21), # /8
            n_flows=self.n_flows,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
            clamp=self.clamp
        ).to(self.device)
        

        self.nf_flow_b  = nf_flow(
            input_size = (128, 56, 42), # /4
            n_flows=self.n_flows,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
            clamp=self.clamp
        ).to(self.device)

        # random.seed(47129)
        self._data_tr_1 = files_sort(glob(os.path.join(self.data_root, 'Train', self.img_type_1, '*.png')))
        self._data_tr_2 = files_sort(glob(os.path.join(self.data_root, 'Train', self.img_type_2, '*.png')))

        print(f'Length of {self.img_type_1}: {len(self._data_tr_1)}, Length of {self.img_type_2}: {len(self._data_tr_2)}')

        self._ds_tr_1 = CTMRDataset(root=self.data_root, img_type='CT', train=1)
        self._ds_tr_2 = CTMRDataset(root=self.data_root, img_type='MRI', train=1)

        self._dl_tr_1 = DataLoader(self._ds_tr_1, batch_size=self.batch_size_tr, shuffle=True, drop_last=True)
        self._dl_tr_2 = DataLoader(self._ds_tr_2, batch_size=self.batch_size_tr, shuffle=True, drop_last=True)

        self._dg_tr_1 = load_data(self._dl_tr_1)
        self._dg_tr_2 = load_data(self._dl_tr_2)

       
        self._ds_source_test = CTMRDataset(root=self.data_root, img_type='CT', train=0)
        self._ds_target_test = CTMRDataset(root=self.data_root, img_type='MRI', train=0)

        self._ds_test = PairDateset(self._ds_source_test, self._ds_target_test)
        self._dl_test = DataLoader(self._ds_test, batch_size=self.batch_size_ts, shuffle=False, drop_last=False)
        self._dg_test = load_data(self._dl_test)

        if self.n_batch == -1:
            # self.n_batch_tr = min(len(self._dl_tr_1), len(self._dl_tr_2))
            self.n_batch_tr = len(self._dl_tr)
        else:
            self.n_batch_tr = self.n_batch

        self.optimizer_t = torch.optim.AdamW(self.nf_flow_t.parameters(), lr=self.lr_t, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.optimizer_b = torch.optim.AdamW(self.nf_flow_b.parameters(), lr=self.lr_b, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        self.lr_schedular_t = PolyLRScheduler(self.optimizer_t, self.lr_t, self.n_epochs)
        self.lr_schedular_b = PolyLRScheduler(self.optimizer_b, self.lr_b, self.n_epochs)

        self.loss_avg_train = AverageCounter()
        self.loss_avg_test = AverageCounter()

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

        self.img_t1_dir = os.path.join(self.output_dir, 'CT')
        os.makedirs(self.img_t1_dir, exist_ok=True)
        self.img_t2_dir = os.path.join(self.output_dir, 'MRI')
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

        if self.ckpt:
            self.nf_flow_b.load_state_dict(torch.load(os.path.join('./mmd_results_' + str(args.task_name), self.ckpt, 'params', f'nf_flow_b_ckpt_{self.ckpt_epoch}.pth')))
            self.nf_flow_t.load_state_dict(torch.load(os.path.join('./mmd_results_' + str(args.task_name), self.ckpt, 'params', f'nf_flow_t_ckpt_{self.ckpt_epoch}.pth')))
            self.logger.append(f'Successfully Loaded NF params of {self.ckpt} with epoch {self.ckpt_epoch}!')

        self.ema_t = EMA(self.nf_flow_t, decay=0.99)
        self.ema_b = EMA(self.nf_flow_b, decay=0.99)

        for epoch in range(self.n_epochs):
            start_time = time.time()
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

        self.nf_flow_t.train()
        self.nf_flow_b.train()

        self.loss_avg_train.restart()
        for _ in range(self.n_batch_tr):
            img_t1 = next(self._dg_tr_1)
            img_t2 = next(self._dg_tr_2)

            img_t1, img_t2 = img_t1.to(self.device), img_t2.to(self.device)

            f_t1_t, f_t1_b = self.vqvae_t1.encode_1(img_t1)
            f_t2_t, f_t2_b = self.vqvae_t2.encode_1(img_t2)

            # print(f_t1_t.shape)
            f_t2_fake, _ = self.nf_flow_t(f_t1_t)
            f_t1_fake, _ = self.nf_flow_t(f_t2_t, rev=True)

            f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1_t), flatten(f_t2_t), flatten(f_t1_fake), flatten(f_t2_fake)
            loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=10) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=10)
            penalty_ot = 0.5 * torch.mean((f_t1_ft - f_t1_fake_ft) ** 2) + 0.5 * torch.mean((f_t2_ft - f_t2_fake_ft) ** 2)
            penalty_l1 = torch.mean(torch.abs(f_t1_ft - f_t1_fake_ft)) + torch.mean(torch.abs(f_t2_ft - f_t2_fake_ft))
            loss_t = loss_mmd + self.weight_ot * penalty_ot

            self.optimizer_t.zero_grad()
            loss_t.backward()
            self.optimizer_t.step()
            self.ema_t.update()

            f_t2_fake, _ = self.nf_flow_b(f_t1_b)
            f_t1_fake, _ = self.nf_flow_b(f_t2_b, rev=True)

            f_t1_ft, f_t2_ft, f_t1_fake_ft, f_t2_fake_ft = flatten(f_t1_b), flatten(f_t2_b), flatten(f_t1_fake), flatten(f_t2_fake)
            loss_mmd = mmd(f_t1_ft, f_t1_fake_ft, kernel_mul=2, kernel_num=10) + mmd(f_t2_ft, f_t2_fake_ft, kernel_mul=2, kernel_num=10)
            penalty_ot = 0.5 * torch.mean((f_t1_ft - f_t1_fake_ft) ** 2) + 0.5 * torch.mean((f_t2_ft - f_t2_fake_ft) ** 2)
            penalty_l1 = torch.mean(torch.abs(f_t1_ft - f_t1_fake_ft)) + torch.mean(torch.abs(f_t2_ft - f_t2_fake_ft))
            loss_b = loss_mmd + self.weight_ot * penalty_ot

            self.optimizer_b.zero_grad()
            loss_b.backward()
            self.optimizer_b.step()
            self.ema_b.update()

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

        # if epoch % 10 == 0:
            img_t1_for_save = torch.concat([img_t1_ae[:10].cpu(), img_t1_fake[:10].cpu()], dim=0)
                # torchvision.utils.save_image(img_t1_for_save, os.path.join(self.img_t1_dir, 'img_' + str(epoch) + '.png'), nrow=self.size_for_show)  
            img_t2_for_save = torch.concat([img_t2_ae[:10].cpu(), img_t2_fake[:10].cpu()], dim=0)


        return self.loss_avg_train.getmean, img_t1_for_save, img_t2_for_save
    
    def test_one_epoch(self, epoch):
        with torch.no_grad():
            self.loss_avg_test.restart()
            self.ema_t.apply_shadow()
            self.ema_b.apply_shadow()
            for i in range(10):
                img_t1, img_t2 = next(self._dg_test)
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
        
        img_t1_for_save = torch.concat([img_t1_ae[:10].cpu(), img_t1_fake[:10].cpu()], dim=0)
        img_t2_for_save = torch.concat([img_t2_ae[:10].cpu(), img_t2_fake[:10].cpu()], dim=0)

        self.ema_t.restore()
        self.ema_b.restore()

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


    
    def train_vqvae(self, img_type='CT'):
        output_dir = 'vqvae_results_' + self.args.task_name
        os.makedirs(output_dir, exist_ok=True)
        ckpt_dir = os.path.join('utils', self.args.task_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        ds = CTMRDataset(root='/home/xiongz/programs/data/Task_003_CT_MRI_Brain_1', img_type=img_type, train=-1)
        print(f'Length of dataset: {ds.__len__()}')
        # ds = PelvisDataset(img_type=img_type)
        dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

        net = VQVAE(
        ).to(self.device)
        # net.load_state_dict(torch.load('utils/Task_008_CT_MR/vqvae_results_ct_res_1.pth'), strict=True)
        # net.load_state_dict(torch.load('utils/Task_008_CT_MR/vqvae_results_ct_res.pth'), strict=True)
        # net.load_state_dict(torch.load('utils/Task_009_Pelvis/vqvae_2_results_mr_res_1.pth'), strict=True)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
        loss_func = torch.nn.MSELoss()

        l_count = AverageCounter()
        for epoch in range(500):
            net.train()
            l_count.restart()
            for data in dl:
                net.zero_grad()
                data = data.to(self.device)
                # data = torch.ones(20, 3, 240, 380).to(self.device)
                # embedding_loss, rec, _ = net(data)
                # recon_loss =  loss_func(rec, data)
                # loss = recon_loss + embedding_loss
                dec, diff = net(data)
                
                recon_loss = loss_func(dec, data)
                latent_loss = diff.mean()

                loss = recon_loss + 2. * latent_loss

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
            print(f'[Epoch: {epoch}], loss: {l_count.getmean:.6f}')
        torch.save(net.state_dict(), os.path.join(ckpt_dir, 'vqvae_results_' + img_type.lower() + '_1.pth'))


    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD for MRI between T1 and T2')
    parser.add_argument('--data_root', type=str, default='/home/xiongz/programs/data/Task_003_CT_MRI_Brain_1')
    parser.add_argument('--task_name', type=str, default='Task_010_CT_MRI_Brain')
    parser.add_argument('--img_type_1', type=str, default='CT')
    parser.add_argument('--img_type_2', type=str, default='MRI')

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

    parser.add_argument('--ckpt_epoch', type=int, default=999)
    parser.add_argument('--ckpt', type=str, default='')

    parser.add_argument('--batch_size_tr', type=int, default=24)
    parser.add_argument('--batch_size_ts', type=int, default=24)
    parser.add_argument('--n_batch', type=int, default=-1)
    parser.add_argument('--unbalanced_num', type=int, default=500)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--use_lr_schedular', default=False, action='store_true')

    parser.add_argument('--weight_ot', default=1e-2, type=float)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_idx)
    mmd_trainer = Trainer(args=args)

    # mmd_trainer.train_ae(img_type='MR')
    # mmd_trainer.train_vqvae(img_type='CT')
    # mmd_trainer.train_vqvae(img_type='MRI')
    mmd_trainer.train_process()
    
    






























