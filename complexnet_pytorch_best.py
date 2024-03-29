from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb, os
import random, sys
import _pickle as cPickle
#import seaborn as sns
import math
import numpy as np
from logger import setup_logger
import complex_new as complex
import JS
from scipy import signal
import scipy.io as sio
from pdb import set_trace as st
logger = setup_logger('JS logger')
logger.info('RadioML')
#os.environ["CUDA_VISIBLE_DEVICES"] ='0,2' 

def calc_next(inputs, kern, stride, outs):
    
    if type(kern) == int:
        dims = int(math.floor((inputs-(kern-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, dims])
    else:
        dims = int(math.floor((inputs-(kern[0]-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, 1])
class ManifoldNetW(nn.Module):
    def __init__(self):
        super(ManifoldNetW, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (5, 1), (5, 1))
        self.proj1 = complex.ReLU4Dsp(20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_1 = complex.ComplexLinearangle4Dmw_outfield(500)
        self.conv_1 = nn.Conv2d(20, 40, (5,1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (5, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.name = "Without Shrinkage"

    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x = self.linear_1((x))
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.mp_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear_3(x)
        return x, torch.zeros(1)

class ManifoldNetW1(nn.Module):
    def __init__(self):
        super(ManifoldNetW1, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (5, 1), (5, 1))
        self.proj1 = complex.ReLU4Dsp(20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_1 = complex.ComplexLinearangle4Dmw_outfield(500)
        self.conv_1 = nn.Conv2d(20, 40, (7, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (3, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.name = "Without Shrinkage"

    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x = self.linear_1((x))
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.mp_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear_3(x)
        return x, torch.zeros(1)

class ManifoldNetRes(nn.Module):
    def __init__(self, num_distr):
        super(ManifoldNetRes, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (5, 1), (5, 1))
        self.proj1 = complex.ReLU4Dsp(20)
        params={'num_classes': 11, 'num_distr': num_distr, 'num_repeat': 2}
        self.SURE = JS.SURE_pure4D(params, calc_next(128, (5, 1), 5, 20), 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 30, (5, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 50, (5, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(50)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Residual Network"
        self.res1=nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2=nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))
        
        
    def make_res_block(self, in_channel, out_channel):    
        res_block = []
        res_block.append(nn.BatchNorm2d(in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), int(out_channel / 4), (3, 1), bias=False, padding=(1, 0)))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), out_channel, (1, 1), bias=False))
        return res_block
    
    
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x, losses = self.SURE(x, labels)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.mp_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss
    
class ManifoldNetComplex(nn.Module):
    def __init__(self, num_distr):
        super(ManifoldNetComplex, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (5, 1), (5, 1))
        self.proj1 = complex.ReLU4Dsp(20)
        params={'num_classes': 11, 'num_distr': num_distr, 'num_repeat': 2}
        self.SURE = JS.SURE_pure4D(params, calc_next(128, (5, 1), 5, 20), 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 40, (5, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (5, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Regular Network"
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x, losses = self.SURE(x, labels)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.mp_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
#         x = self.dropout(x)
        x = self.linear_3(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss
class Test(nn.Module):
    def __init__(self, num_distr):
        super(Test, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (5, 1), (5, 1))
        self.proj1 = complex.ReLU4Dsp(20)
        params={'num_classes': 11, 'num_distr': num_distr, 'num_repeat': 2}
        self.SURE = JS.SURE_pure4D(params, calc_next(128, (5, 1), 5, 20), 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 40, (5, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (5, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Regular Network"
    def forward(self, x, labels=None):
        x0 = x
        x1 = self.complex_conv1(x)
        x = self.proj1(x1)
        x2, losses = self.SURE(x, labels)
        x = self.relu(x2)
        x3 = self.conv_1(x)
        x = self.bn_1(x3)
        x = self.relu(x)
        x = self.mp_1(x)
        x4 = self.conv_2(x)
        x = self.bn_2(x4)
        x = self.relu(x)
        x = self.mp_2(x)
        x5 = self.conv_3(x)
        x = self.relu(x5)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return self.SURE.miu, x0, x1, x2, x3, x4, x5, x, res_loss    
    
class STFT(nn.Module):
    def __init__(self, num_distr):
        super(STFT, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (3, 3), (1, 3))
        self.proj1 = complex.ReLU4Dsp(20)
        params={'num_classes': 11, 'num_distr': num_distr, 'num_repeat': 2}
        self.SURE = JS.SURE_pure4D(params, torch.Size([2, 20, 6, 11]), 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 40, (3, 3))
        self.mp_1 = nn.MaxPool2d((1,2))
        self.conv_2 = nn.Conv2d(40, 60, (3, 3))
#         self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (2, 2))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Regular Network STFT"
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
#         print(x.shape)
        x, losses = self.SURE(x, labels)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
#         x = self.mp_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss
    
class STFT2(nn.Module):
    def __init__(self, num_distr):
        super(STFT2, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffangle4Dxy(1, 20, (1, 3), (1, 3))
        self.proj1 = complex.ReLU4Dsp(20)
        params={'num_classes': 11, 'num_distr': num_distr, 'num_repeat': 2}
        self.SURE = JS.SURE_pure4D(params, torch.Size([2, 20, 8, 11]), 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 40, (3, 3))
        self.mp_1 = nn.MaxPool2d((1,2))
        self.conv_2 = nn.Conv2d(40, 60, (3, 3))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (2, 2))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Regular Network STFT"
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
#         print(x.shape)
        x, losses = self.SURE(x, labels)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.mp_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dropout_ratio = 0.5
        self.conv1 = nn.Conv2d(1, 256, padding= [0,2], kernel_size=[1,3])
        self.dropout2d = nn.Dropout2d(dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)
        self.conv2 = nn.Conv2d(256, 80, padding= [0,2], kernel_size=[2,3]) 
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10560, 256)
        self.fc2 = nn.Linear(256, 11)
        self.name='Baseline'

    def forward(self, x, labels=None):
        x = self.dropout2d(F.relu(self.conv1(x), 2))
        x = self.dropout2d(F.relu(self.conv2(x), 2))
#         print(x.shape)
        x = x.view(-1, 10560)
        x = self.dropout(F.relu(self.fc1(x)))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), torch.zeros(1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)
    for batch_idx, (data, target) in enumerate(train_loader):
        #scheduler.step()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, losses = model(data, target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        
       
        sums = torch.sum(losses)
        loss = loss + sums 
        
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def eval_train(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = []
    real_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, losses = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            
        
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
             
    test_loss /= len(test_loader.dataset)
    logger.info('\nTraining set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))           
            
def test(args, model, device, test_loader, lbl, snrs, test_idx, best_acc, save_path):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = []
    real_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, losses = model(data)
            #output = model.linear(model.proj3(model.complex_conv3(model.proj2(model.complex_conv2(model.proj1(model.complex_conv1(data)))))))
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            pred_all.append(np.array(pred.cpu()))
            real_all.append(np.array(target.cpu()))
            correct += pred.eq(target.view_as(pred)).sum().item()
    pred_all = np.squeeze( np.concatenate(pred_all) )
    real_all = np.concatenate(real_all)
    
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        pred_i = pred_all[np.where(np.array(test_SNRs)==snr)]
        real_i = real_all[np.where(np.array(test_SNRs)==snr)]
        logger.info('SNR ' +str(snr)+' test accuracy: '+str(100. * np.mean(pred_i==real_i) )+'%.')
             
    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if 100. * correct / len(test_loader.dataset) > best_acc:
        best_acc = 100. * correct / len(test_loader.dataset)
        if save_path != None:
            os.remove(save_path)
        save_path = os.path.join('./save/', 'RadioML-[{acc}]-[{name}].ckpt'.format(acc = best_acc, name=model.name))
        torch.save(model.state_dict(), save_path)
        logger.info('Saved model checkpoints into {}...'.format(save_path))
    return save_path, best_acc

def to_polarangle(X):
    M = np.linalg.norm(X,axis=1)
    T = np.arctan2(X[:,1,:],X[:,0,:])
    T = np.expand_dims(T,axis=1)
    Y = np.expand_dims(np.expand_dims(np.concatenate((T,np.expand_dims(M,axis=1)),axis=1),axis=3),axis=2)
    return Y

def to_polar4Dnlog(X):
    M = np.log(np.linalg.norm(X, axis=1)+0.001)
    T = np.arctan2(X[:,1,:],X[:,0,:])
    MT = np.amax(T)
    mT = np.amin(T)
    MM = np.amax(M)
    mM = np.amin(M)
    #T = np.expand_dims(2*((T-mT)/(MT-mT))-1.0+0.00001,axis=1)
    #M = ((M-mM)/(MM-mM))+0.00001
    Y = np.expand_dims(np.expand_dims(np.concatenate((np.expand_dims(T,axis=1),np.expand_dims(M,axis=1),X),axis=1),axis=3),axis=2)
    return Y


def to_polar4D_stft(X):
    M = np.linalg.norm(X, axis=1)
    T = np.arctan2(X[:,1,...],X[:,0,...])
    MT = np.amax(T)
    mT = np.amin(T)
    MM = np.amax(M)
    mM = np.amin(M)
    #T = np.expand_dims(2*((T-mT)/(MT-mT))-1.0+0.00001,axis=1)
    #M = ((M-mM)/(MM-mM))+0.00001
    Y = np.expand_dims(np.concatenate((np.expand_dims(T,axis=1),np.expand_dims(M,axis=1),X),axis=1),axis=2)
    return Y



def to_polar4D(X):
    M = np.linalg.norm(X, axis=1)
    T = np.arctan2(X[:,1,:],X[:,0,:])
    MT = np.amax(T)
    mT = np.amin(T)
    MM = np.amax(M)
    mM = np.amin(M)
    #T = np.expand_dims(2*((T-mT)/(MT-mT))-1.0+0.00001,axis=1)
    #M = ((M-mM)/(MM-mM))+0.00001
    Y = np.expand_dims(np.expand_dims(np.concatenate((np.expand_dims(T,axis=1),np.expand_dims(M,axis=1),X),axis=1),axis=3),axis=2)
    return Y

def to_polar(X):
    M = np.linalg.norm(X,axis=1)
    T = np.arctan2(X[:,1,:],X[:,0,:]) 
    C = np.cos(T+1e-5)
    S = np.sin(T)
    M = np.expand_dims(M,axis=1)
    C = np.expand_dims(C,axis=1)
    S = np.expand_dims(S,axis=1)
    Y = np.expand_dims(np.expand_dims(np.concatenate((C,S,-S,C,M),axis=1),axis=3),axis=2)
    return Y


def data_prep(path, batch_size):
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(list(yy))),yy] = 1
        return yy1
    
    Xd = cPickle.load(open(path,'rb'), encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
    
    train_idx = np.load("../data/train_idx.npy")
    test_idx = np.load("../data/test_idx.npy")
    # randomly split train/ test
    '''np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    np.save("data/train_idx", train_idx)
    np.save("data/test_idx", test_idx)'''
    X_train = X[train_idx]
    X_test = X[test_idx]
    X_train = (X_train - np.mean(X_train) ) / np.std(X_train)
    X_test = (X_test - np.mean(X_test) ) / np.std(X_test)
    # need to write normalization here

    X_train = to_polar4D(X_train)
    X_test = to_polar4D(X_test)
    
#     X_train = np.expand_dims(X_train, axis=1)
#     X_test = np.expand_dims(X_test, axis=1)
    
    #Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    #Y_test = to_onehot( list(map(lambda x: mods.index(lbl[x][0]), test_idx)) )
    Y_train = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), train_idx)) )
    Y_test = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), test_idx)) )
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy (Y_train).type(torch.LongTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = True)
    test_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle = False)
    
    return train_loader_dataset, test_loader_dataset, lbl, snrs, test_idx


def data_prep_visualize(path, batch_size):
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(list(yy))),yy] = 1
        return yy1

    Xd = cPickle.load(open('../data/RML2016.10a_dict.pkl','rb'), encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    seen = []
    
    for i in range(len(mods)):
        for snr in snrs:
            if snr == snrs[-1]:
                if mods[i] not in seen:
                    X.append(np.expand_dims(Xd[(mods[i],snr)][0], axis=0))
                    seen.append(mods[i])
                    lbl.append((mods[i],snr))
    X = np.vstack(X)

    Y = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), np.arange(11))) )

    
    X = (X - np.mean(X) ) / np.std(X)
    
    X = to_polar4D(X)
    
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy (Y).type(torch.LongTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle = False)
    
    return train_loader_dataset

def visualize(model, data_generator):
    kk=1
    save_dict = {}
    for it,(local_batch, local_labels) in enumerate(data_generator):
        batch = torch.tensor(local_batch, requires_grad=True).cuda()
        sure, x0, x1, x2, x3, x4, x5, x, res_loss = model(batch, None)
#         label = local_labels.item()
#         save_dict['c'+str(label)] = []
        
#         img = x0[0, :2, 0, :, 0].cpu().detach().numpy()
#         img = img.transpose((1, 0))
        
#         #[128, 1]
#         phase = np.expand_dims(img[:, 0] / np.pi, axis=1)
        
        
#         save_dict['c'+str(label)].append(phase)
        
#         #[128, 1]
#         mag = np.expand_dims(img[:, 1], axis=1)
        
        
# #         img = np.expand_dims(phase, axis=2)
# #         img[:, :, 0] = mag
#         #[128, 3]
#         mag = np.insert(mag, 1, 1, axis=1)
#         mag = np.insert(mag, 2, 1, axis=1)
        
#         save_dict['c'+str(label)].append(mag)
#         kk+=1

#         img1 = x1[0, 2:, 0:10, :, 0].cpu().detach().numpy()
        
#         imshape = img1.shape
#         img1 = img1.transpose((1, 2, 0))
        
#         M = np.linalg.norm(img1, axis=2)
#         T = np.arctan2(img1[:,:,1],img1[:,:,0]) / np.pi
        
#         phase = np.expand_dims(T, axis=2)
#         mag = np.expand_dims(M, axis=2)
#         min_0 = np.min(phase, keepdims=True)
#         max_0 = np.max(phase, keepdims=True)
#         min_2 = np.min(mag, keepdims=True)
#         max_2 = np.max(mag, keepdims=True)
#         phase = (phase - min_0)/ (max_0-min_0)
#         mag = (mag - min_2)/ (max_2-min_2)
        
#         mag = np.insert(mag, 1, 1, axis=2)
#         mag = np.insert(mag, 2, 1, axis=2)
        
#         phase = phase.transpose((1, 2, 0))
#         mag = mag.transpose((1, 2, 0))
     
#         save_dict['c'+str(label)].append(phase)
#         save_dict['c'+str(label)].append(mag)
#         kk+=1

#         img2 = x2[0, 0:10, :, :].cpu().detach().numpy()
#         img2 = img2.transpose((1, 2, 0))
#         save_dict['c'+str(label)].append(img2)
        
#         kk+=1

#         img3 = x3[0, 0:10, :, :].cpu().detach().numpy()
#         img3 = img3.transpose((1, 2, 0))

#         save_dict['c'+str(label)].append(img3)

#         kk+=1
        
#         img4 = x4[0, 0:10, :, :].cpu().detach().numpy()
#         imshape = img4.shape

#         save_dict['c'+str(label)].append(img4)

#         kk+=1

#         img5 = x5[0, :, :, :].cpu().detach().numpy().reshape(x5.shape[1], 1)
#         save_dict['c'+str(label)].append(img5)

#         kk+=1
    for i in range(5):
        img6 = sure[i, :, 0:10, :, 0].cpu().detach().numpy()
        
        imshape = img6.shape
        img6 = img6.transpose((1, 2, 0))
        
        M = np.linalg.norm(img6, axis=2)
        T = np.arctan2(img6[:,:,1],img6[:,:,0]) / np.pi
        
        phase = np.expand_dims(T, axis=2)
        mag = np.expand_dims(M, axis=2)
        min_0 = np.min(phase, keepdims=True)
        max_0 = np.max(phase, keepdims=True)
        min_2 = np.min(mag, keepdims=True)
        max_2 = np.max(mag, keepdims=True)
        phase = (phase - min_0)/ (max_0-min_0)
        mag = (mag - min_2)/ (max_2-min_2)
        
        mag = np.insert(mag, 1, 1, axis=2)
        mag = np.insert(mag, 2, 1, axis=2)
        
        phase = phase.transpose((1, 2, 0))
        mag = mag.transpose((1, 2, 0))
     
        save_dict['d0_'+str(i)]=phase
        save_dict['d1_'+str(i)]=mag
        
        
    sio.savemat('data_new.mat', save_dict)


def data_prep1(path, batch_size):
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(list(yy))),yy] = 1
        return yy1
    
    Xd = cPickle.load(open(path,'rb'), encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
    
    train_idx = np.load("../data/train_idx.npy")
    test_idx = np.load("../data/test_idx.npy")
    # randomly split train/ test
    '''np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    np.save("data/train_idx", train_idx)
    np.save("data/test_idx", test_idx)'''
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    
    x_train = X_train[:, 0, :] + 1j*X_train[:, 1, :]
    f, t, Zxx = signal.stft(x_train, fs=0.5, window='blackman', nperseg=8, noverlap=4)
#     print(Zxx.shape)
    
    #[B, 1, H, W]
    Zxx_real = np.expand_dims(Zxx.real, axis=1)
    Zxx_imag = np.expand_dims(Zxx.imag, axis=1)
    
    Zxx = np.concatenate((Zxx_real, Zxx_imag), axis=1)
    Zxx = (Zxx - np.mean(Zxx) ) / np.std(Zxx)
    
    
    x_test = X_test[:, 0, :] + 1j*X_test[:, 1, :]
    f_, t_, Zxx_ = signal.stft(x_test, fs=0.5, window='blackman', nperseg=8, noverlap=4)
    
    #[B, 1, H, W]
    Zxx_real_test = np.expand_dims(Zxx_.real, axis=1)
    Zxx_imag_test = np.expand_dims(Zxx_.imag, axis=1)
    
    Zxx_ = np.concatenate((Zxx_real_test, Zxx_imag_test), axis=1)
    Zxx_ = (Zxx_ - np.mean(Zxx_) ) / np.std(Zxx_)

    X_train = to_polar4D_stft(Zxx)
    X_test = to_polar4D_stft(Zxx_)
    
    #Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    #Y_test = to_onehot( list(map(lambda x: mods.index(lbl[x][0]), test_idx)) )
    Y_train = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), train_idx)) )
    Y_test = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), test_idx)) )
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy (Y_train).type(torch.LongTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = True)
    test_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle = False)
    
    return train_loader_dataset, test_loader_dataset, lbl, snrs, test_idx
def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch RadioML Example') #400 and 0.001
    parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batchsize', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Adam momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
#     model = Test(5).to(device)
#     save_path = os.path.join('./save/', 'RadioML-[53.21636363636364].ckpt')
#     model.load_state_dict(torch.load(save_path))
#     train_gen = data_prep_visualize("../data/RML2016.10a_dict.pkl", 100)
#     visualize(model, train_gen)
#     exit()
        
#     num_distr = [5, 8, 10, 3]
#     batches = [300, 50, 100, 200, 400, 500, 800, 1000, 30, 80, 300]
#     lrs = [0.04, 0.08]
    
    num_distr = [3, 4, 5, 8, 11]
    batches = [200, 400, 500, 200, 100, 800, 50]
    lrs = [0.03, 0.02, 0.01, 0.03, 0.008, 0.005, 0.05]
    
    for n in num_distr:
    #model = Net().to(device)
        model = ManifoldNetComplex(n).to(device)
        logger.info(model.name)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("##########")
        logger.info("Model Parameters: "+str(params))
        logger.info("Number of distributions: "+str(n))

        for b in batches:
            logger.info("##########")
            logger.info("Batch size: "+str(b))
            train_loader, test_loader, lbl, snrs, test_idx = data_prep("../data/RML2016.10a_dict.pkl", b)

            for lr_ in lrs:
                optimizer = optim.Adam(model.parameters(), lr=lr_, eps=1e-3, amsgrad=True )#momentum=args.momentum)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
                logger.info("##########")
                logger.info("Model Parameters: "+str(params))
                logger.info("Number of distributions: "+str(n))
                logger.info("##########")
                logger.info("Batch size: "+str(b))
                logger.info("###########")
                logger.info('Learning Rate: '+str(lr_))
                best_acc = 0
                save_path = None

                for epoch in range(1, args.epochs + 1):
                    train(args, model, device, train_loader, optimizer, epoch)
                    
                    eval_train(args, model, device, train_loader)
                    save_path, best_acc = test(args, model, device, test_loader, lbl, snrs, test_idx, best_acc, save_path)
                    logger.info(str([n, b, lr_]))
                    if epoch > 60:
                        scheduler.step()
                model = ManifoldNetComplex(n).to(device)
#     for n in num_distr:
#     #model = Net().to(device)
#         model = STFT(n).to(device)
#         logger.info(model.name)
#         model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         logger.info("##########")
#         logger.info("Model Parameters: "+str(params))
#         logger.info("Number of distributions: "+str(n))

#         for b in batches:
#             logger.info("##########")
#             logger.info("Batch size: "+str(b))
#             train_loader, test_loader, lbl, snrs, test_idx = data_prep1("../data/RML2016.10a_dict.pkl", b)

#             for lr_ in lrs:
#                 optimizer = optim.Adam(model.parameters(), lr=lr_, eps=1e-3, amsgrad=True )#momentum=args.momentum)
#                 logger.info("##########")
#                 logger.info("Model Parameters: "+str(params))
#                 logger.info("Number of distributions: "+str(n))
#                 logger.info("##########")
#                 logger.info("Batch size: "+str(b))
#                 logger.info("###########")
#                 logger.info('Learning Rate: '+str(lr_))
#                 best_acc = 0
#                 save_path = None

#                 for epoch in range(1, args.epochs + 1):
#                     train(args, model, device, train_loader, optimizer, epoch)
#                     eval_train(args, model, device, train_loader)
#                     save_path, best_acc = test(args, model, device, test_loader, lbl, snrs, test_idx, best_acc, save_path)
#                     logger.info(str([n, b, lr_]))

if __name__ == '__main__':
    main()
