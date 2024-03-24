import os
import torch
from torch import nn
import math
import argparse

import numpy as np
from sklearn import datasets

import matplotlib.pyplot as plt
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.distributions

from torch.utils.data import DataLoader
from logger import Logger
import pathlib
import json
import datetime
# from torch.optim import lr_scheduler


def make_circles_ssl(n_samples, seed=0):
    
    np.random.seed(seed)
    data, y = datasets.make_circles(n_samples=n_samples, noise=.05, factor=0.4) # [0].astype(np.float32)
    data = data.astype(np.float32)
    
    return data

def make_moons_ssl(n_samples, seed=0):

    np.random.seed(seed)
    # n_samples = 2000
    data, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    data = data.astype(np.float32)
    
    return data

def make_squaredata_ssl(n_samples=2000, seed=0):#, label_ratio=0.05):

    np.random.seed(seed)
    data = np.random.rand(n_samples, 2)
    data = data.astype(np.float32)

    return data

def make_multigaussian_ssl(n_samples, n_gauss=8, start_angle=0., radius=2., var=0.2, seed=0):
    data = np.zeros((n_samples * n_gauss, 2))
    means = np.zeros((n_gauss, 2))
    np.random.seed(seed)
    for t in range(n_gauss):
        means[t][0] = radius * math.cos(2 * t * math.pi / n_gauss + start_angle)
        means[t][1] = radius * math.sin(2 * t * math.pi / n_gauss + start_angle)
        data[t * n_samples: (t + 1) * n_samples] = means[t] + var * np.random.randn(n_samples, 2)
    
    return np.float32(data)

def make_single_gauss(n_samples, mean, cov, seed=0):
    np.random.seed(seed)
    return np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples).astype(np.float32)

class ToyDateset(torch.utils.data.Dataset):
    def __init__(self, source_data, target_data):
        self.x1 = source_data
        self.x2 = target_data
    def __getitem__(self, index):
        x1_sample = self.x1[index]
        x2_sample = self.x2[index]
        return torch.from_numpy(x1_sample), torch.from_numpy(x2_sample)
    def __len__(self):
        return len(self.x1)

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

def ed_kernel(source, target):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    return -(L2_distance)

def mmd(source, target, kernel_type='gaussian_kernel', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    assert kernel_type in ['gaussian_kernel', 'ed_kernel']

    batch_size = int(source.size()[0])
    if kernel_type == 'gaussian_kernel':
        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    elif kernel_type == 'ed_kernel':
        kernels = ed_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def subnet_fc(mid_dim=256):
    def fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, mid_dim), nn.ReLU(),
                            nn.Linear(mid_dim,  dims_out))
    return fc

def nf_flow(input_size, args):
    n_dim = input_size
    flow = Ff.SequenceINN(*n_dim)
    for _ in range(args.n_flows):
        flow.append(
            Fm.AllInOneBlock, 
            subnet_constructor=subnet_fc(mid_dim=args.mid_dim), 
            affine_clamping=2.0, 
            permute_soft=True
        )
    return flow

def nf_head(input_dim, n_coupling_blocks=8):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': 3, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': 256, 'dropout': 0.0}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder

def build_dataset(datatype, args):
    if datatype == 'moons':
        data = make_moons_ssl(n_samples=args.n_samples, seed=args.data_seed)
    elif datatype == 'circles':
        data = make_circles_ssl(n_samples=args.n_samples, seed=args.data_seed)
    elif datatype == 'gauss':
        data = make_multigaussian_ssl(n_samples=args.n_samples, seed=args.seed)
    return data
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mmd training')
    parser.add_argument('--training_name', type=str, default='m2c')
    parser.add_argument('--source_data', type=str, default='moons')
    parser.add_argument('--target_data', type=str, default='circles')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--data_seed', type=int, default=2023)
    parser.add_argument('--lr_init', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_flows', type=int, default=8)
    parser.add_argument('--mid_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200)

    parser.add_argument('--mmd_kernel_mul', type=int, default=2)
    parser.add_argument('--mmd_kernel_num', type=int, default=10)

    parser.add_argument('--use_ot', action='store_true')
    parser.add_argument('--ot_weight', type=float, default=5e-4)


    args = parser.parse_args()
    # print(args)
    # exit()
    device = 'cuda:7'

    output_dir = './mmd_results_' + str(args.training_name)
    output_dir = (
        pathlib.Path(output_dir).resolve().joinpath(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    argsDict = args.__dict__
    args_path = str(output_dir) + '/args.json'

    json_str = json.dumps(argsDict, indent=4)
    with open(args_path, 'w') as json_file:
        json_file.write(json_str)
    logger = Logger(str(output_dir) +'/train.log')

    
    source_data = build_dataset(datatype=args.source_data, args=args)
    target_data = build_dataset(datatype=args.target_data, args=args)
    
    train_dataset = ToyDateset(source_data=source_data, target_data=target_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    n_dim = (2, )
    flow = nf_flow(input_size=n_dim, args=args)
    # flow = nf_head(input_dim=2)
    flow = flow.to(device)

    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(flow.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=5e-3,total_steps=100)

    for epoch in range(1, args.n_epochs + 1):
        loss_record = 0. 
        mmd_record = 0.
        ot_record = 0.

        cnt = 0
        for x1, x2 in train_dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            x2_fake, _ = flow(x1)
            x1_fake, _ = flow(x2, rev=True)

            loss_mmd = mmd(x2_fake, x2, kernel_mul=args.mmd_kernel_mul, kernel_num=args.mmd_kernel_num) + mmd(x1_fake, x1, kernel_mul=args.mmd_kernel_mul, kernel_num=args.mmd_kernel_num) 
            # penalty = ot_penalty(flow, x1, lbda=1e-2)
            # penalty_ot = torch.sum((x1 - x2_fake) ** 2)
            penalty_ot = 0.5 * torch.mean((x1 - x2_fake) ** 2) + 0.5 * torch.mean((x2 - x1_fake) ** 2)

            # loss = torch.mean(0.5 * torch.sum(x2_fake ** 2, dim=1) - log_det_jac)

            if args.use_ot:
                loss = loss_mmd + args.ot_weight * penalty_ot
            else:
                loss = loss_mmd
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
        
            loss_record += loss.item()
            mmd_record += loss_mmd.item()
            ot_record += penalty_ot.item()
            
            cnt += 1

        loss_avg = loss_record / cnt
        mmd_avg = mmd_record / cnt
        ot_avg = ot_record / cnt
        
        logger.append('epoch: {}, loss: {:.6f} loss_mmd: {:.6f}, ot: {:.6f}'.format(epoch, loss_avg, mmd_avg, ot_avg))
    torch.save(flow.state_dict(), str(output_dir) + '/net_params.pth')

    flow.eval()
    data_1 = make_moons_ssl(n_samples=1000, seed=2023)
    data_2 = make_circles_ssl(n_samples=1000, seed=2023)
    # data_1 = make_single_gauss(n_samples=1000, mean=mean1, cov=cov1, seed=2023)
    # data_2 = make_single_gauss(n_samples=1000, mean=mean2, cov=cov2, seed=2023)
    with torch.no_grad():
        fake_data_2, _ = flow(torch.from_numpy(data_1).to(device))
        fake_data_1, _ = flow(torch.from_numpy(data_2).to(device), rev=True)
    fake_data_1, fake_data_2 = fake_data_1.cpu(), fake_data_2.cpu()

    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.scatter(data_1[:,0], data_1[:,1], s=3, label='original')
    plt.scatter(fake_data_2.numpy()[:,0], fake_data_2.numpy()[:,1], s=3, label='generated')
    plt.xlim([-12,4])
    plt.ylim([-4,4])
    plt.legend()
    for i in range(0, 1000, 10):
        plt.plot([data_1[i,0], fake_data_2.numpy()[i,0]], [data_1[i,1], fake_data_2.numpy()[i,1]], c='green', linewidth=0.3)
    plt.subplot(2,2,2)
    plt.scatter(data_2[:,0], data_2[:,1], s=3, label='original')
    plt.scatter(fake_data_1.numpy()[:,0], fake_data_1.numpy()[:,1], s=3, label='generated')
    plt.xlim([-12,4])
    plt.ylim([-4,4])
    plt.legend()
    for i in range(0, 1000, 10):
        plt.plot([data_2[i,0], fake_data_1.numpy()[i,0]], [data_2[i,1], fake_data_1.numpy()[i,1]], c='green', linewidth=0.3)
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.subplot(2,2,3)
    # plt.scatter(data_2[:,0], data_2[:,1], s=3)
    plt.scatter(fake_data_2.numpy()[:,0], fake_data_2.numpy()[:,1], s=3)
    plt.xlim([-12,4])
    plt.ylim([-4,4])
    plt.subplot(2,2,4)
    # plt.scatter(data_1[:,0], data_1[:,1], s=3)
    plt.scatter(fake_data_1.numpy()[:,0], fake_data_1.numpy()[:,1], s=3)
    plt.xlim([-12,4])
    plt.ylim([-4,4])
    plt.savefig(str(output_dir) + '/figure.png')
    
