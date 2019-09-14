import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import complex_new as complex
import math
from JS import SURE
from JS import SURE_pure
from pdb import set_trace as st



def calc_next(inputs, kern, stride, outs):
    
    if type(kern) == int:
        dims = int(math.floor((inputs-(kern-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, dims])
    else:
        dims = int(math.floor((inputs-(kern[0]-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, 1])
        


class ManifoldNetComplex(nn.Module):
    def __init__(self, num_classes, num_distr, num_repeat):
        super(ManifoldNetComplex, self).__init__()
        params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 5, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(5, 10, (5, 5), (2, 2))
        self.complex_conv3 = complex.ComplexConv2Deffgroup(10, 20, (5, 5), (2, 2))
        self.complex_conv4 = complex.ComplexConv2Deffgroup(20, 40, (5, 5), (2, 2))
        self.SURE = SURE(params, 40, 80, (3, 3), (1, 1), calc_next(3, 3, 1, 5))
        self.complex_bn = complex.ComplexBN()
        self.proj1 = complex.manifoldReLUv2angle(5) #complex.ReLU4Dsp(20)
        self.proj2 = complex.manifoldReLUv2angle(10) #complex.ReLU4Dsp(40)
        self.proj3 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(80)
        self.proj4 = complex.manifoldReLUv2angle(40) #complex.ReLU4Dsp(80)
        self.proj5 = complex.manifoldReLUv2angle(80) #complex.ReLU4Dsp(80)
        self.linear1 = nn.Linear(80, 30)
        self.linear2 = nn.Linear(30, 11)
        self.relu = nn.ReLU()
        self.name='complex only'
    
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x = self.complex_conv2(x)
        x = self.proj2(x)
        x = self.complex_conv3(x)
        x = self.proj3(x)
        x = self.complex_conv4(x)
        x = self.proj4(x)
        x, losses = self.SURE(x, labels)
        x = self.linear1(x.squeeze(-1).squeeze(-1))
        x = self.relu(x)
        x = self.linear2(x)
        return x, losses
  
    def clear_weights(self):
        self.SURE.clear_LE()
        
        
#This is the network used for 10-class classification task
class ManifoldNetComplex1(nn.Module):
    def __init__(self, num_classes, num_distr, num_repeat):
        super(ManifoldNetComplex1, self).__init__()
        params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (5, 5)) #20, 20
        self.SURE = SURE_pure(params, calc_next(100, 5, 5, 20), 20) 
#         self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
#         self.SURE = SURE(params, 20, 20, (5, 5), (1, 1), calc_next(22, 5, 1, 20))
        self.name = 'complex+standardCNN'
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
#         self.conv_1 = nn.Conv2d(20, 30, (5,5), (3,3))
#         self.mp_1 = nn.MaxPool2d((2,2))
#         self.conv_2 = nn.Conv2d(30, 40, (2,2))
#         self.bn_1 = nn.BatchNorm2d(30)
#         self.bn_2 = nn.BatchNorm2d(40)
#         self.linear_2 = nn.Linear(40, 20)
#         self.linear_4 = nn.Linear(20, 11)

#         self.conv_1 = nn.Conv2d(20, 30, (5,5))
        self.conv_1 = nn.Conv2d(20, 40, (5,5))
        self.mp_1 = nn.MaxPool2d((2,2))
        self.conv_2 = nn.Conv2d(40, 60, (3,3))
        self.mp_2 = nn.MaxPool2d((2,2))
        self.conv_3 = nn.Conv2d(60, 80, (3,3))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_4 = nn.Linear(40, 11)
        
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj2(x)
#         x = self.complex_conv2(x)
#         x = self.proj2(x)
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
        x = self.linear_4(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss
    
    def clear_weights(self):
        self.SURE.clear_LE()

class Test(nn.Module):
    def __init__(self, num_classes, num_distr, num_repeat):
        super(Test, self).__init__()
        params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (5, 5)) #20, 20
        self.SURE = SURE_pure(params, calc_next(100, 5, 5, 20), 20)
        self.name = 'complex+standardCNN'
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv_1 = nn.Conv2d(20, 40, (5,5))
        self.mp_1 = nn.MaxPool2d((2,2))
        self.conv_2 = nn.Conv2d(40, 60, (3,3))
        self.mp_2 = nn.MaxPool2d((2,2))
        self.conv_3 = nn.Conv2d(60, 80, (3,3))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_4 = nn.Linear(40, 11)
        
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
    def forward(self, x, labels=None):
        x0=x
        x1 = self.complex_conv1(x)
        x = self.proj2(x1)
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
#         x = self.bn_3(x5)
        x = self.relu(x5)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x0, x1, x2, x3, x4, x5, x, res_loss
    
    def clear_weights(self):
        self.SURE.clear_LE()
        
# class ManifoldNetComplex1(nn.Module):
#     def __init__(self, num_classes, num_distr, num_repeat):
#         super(ManifoldNetComplex1, self).__init__()
#         params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
#         self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (2, 2)) #20, 20
# #         self.SURE = SURE_pure(params, calc_next(100, 5, 5, 20), 20) 
#         self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
#         self.SURE = SURE(params, 20, 20, (5, 5), (1, 1), calc_next(22, 5, 1, 20))
#         self.name = 'complex+standardCNN'
#         self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.conv_1 = nn.Conv2d(20, 30, (5,5))
#         self.mp_1 = nn.MaxPool2d((2,2))
#         self.conv_2 = nn.Conv2d(30, 40, (3,3))
#         self.bn_1 = nn.BatchNorm2d(30)
#         self.bn_2 = nn.BatchNorm2d(40)
#         self.linear_2 = nn.Linear(50, 20)
#         self.linear_4 = nn.Linear(20, 11)

# # #         self.conv_1 = nn.Conv2d(20, 30, (5,5))
# #         self.conv_1 = nn.Conv2d(20, 40, (3,3))
# #         self.mp_1 = nn.MaxPool2d((2,2))
# #         self.conv_2 = nn.Conv2d(40, 60, (3,3))
#         self.mp_2 = nn.MaxPool2d((2,2))
#         self.conv_3 = nn.Conv2d(40, 50, (2,2))
# #         self.bn_1 = nn.BatchNorm2d(40)
# #         self.bn_2 = nn.BatchNorm2d(60)
#         self.bn_3 = nn.BatchNorm2d(50)
# #         self.linear_2 = nn.Linear(80, 40)
# #         self.linear_4 = nn.Linear(40, 11)
        
#         self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
#     def forward(self, x, labels=None):
#         x = self.complex_conv1(x)
#         x = self.proj2(x)
#         x = self.complex_conv2(x)
#         x = self.proj2(x)
#         x, losses = self.SURE(x, labels)
#         x = self.relu(x)
#         x = self.conv_1(x)
#         x = self.bn_1(x)
#         x = self.relu(x)
#         x = self.mp_1(x)
#         x = self.conv_2(x)
#         x = self.bn_2(x)
#         x = self.relu(x)
#         x = self.mp_2(x)
#         x = self.conv_3(x)
#         x = self.bn_3(x)
#         x = self.relu(x)
#         x = x.squeeze(-1).squeeze(-1)
#         x = self.linear_2(x)
#         x = self.relu(x)
#         x = self.linear_4(x)
#         res_loss = 0
#         if losses is not None:
#             res_loss = losses * (self.loss_weight ** 2)
#         return x, res_loss
    
#     def clear_weights(self):
#         self.SURE.clear_LE()
        
class ManifoldNetR(nn.Module):
    
    def __init__(self, num_classes, num_distr, num_repeat):
        super(ManifoldNetR, self).__init__()
        params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 40, (5, 1), (2, 1))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(40, 40, (5, 1), (2, 1))
        self.SURE = SURE(params, 40, 40, (5, 1), (1, 1), calc_next(29, (5, 1), 1, 20))
        
        self.proj2 = complex.manifoldReLUv2angle(40) #complex.ReLU4Dsp(40)
        self.relu = nn.ReLU()

        self.conv_1 = nn.Conv2d(40, 50, (5,1))
        self.mp_1 = nn.MaxPool2d((3,1))
        self.conv_2 = nn.Conv2d(50, 60, (5,1))
        self.conv_3 = nn.Conv2d(60, 80, (3,1))
        self.bn_1 = nn.BatchNorm2d(50)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_4 = nn.Linear(40, 11)
        self.name = 'Complex Radio'
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj2(x)
        x = self.complex_conv2(x)
        x = self.proj2(x)
        
        x, losses = self.SURE(x, labels)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss
   

    
    def clear_weights(self):
        self.SURE.clear_LE()
        return 0

class ManifoldNetRes(nn.Module):
    def __init__(self, num_classes, num_distr, num_repeat):
        super(ManifoldNetRes, self).__init__()
        params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 1), (2, 1))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 1), (2, 1))
        self.complex_conv3 = complex.ComplexConv2Deffgroup(20, 20, (5, 1), (1, 1))
        self.SURE = SURE_pure(params, calc_next(29, (5, 1), 1, 20), 20)
        
        self.complex_res1 = complex.ResidualLayer(20, 20, 20, (5, 1), (2, 1))
        self.complex_res2 = complex.ResidualLayer(20, 20, 20, (5, 1), (1, 1))
        
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
        self.relu = nn.ReLU()

        self.conv_1 = nn.Conv2d(20, 30, (5,1))
        self.mp_1 = nn.MaxPool2d((3,1))
        self.conv_2 = nn.Conv2d(40, 50, (5,1))
        self.conv_3 = nn.Conv2d(60, 70, (3,1))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(50)
        self.bn_3 = nn.BatchNorm2d(70)
        self.linear_2 = nn.Linear(70, 40)
        self.linear_4 = nn.Linear(40, 11)
        self.name = 'Residual complex for radio'
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        
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
        res_block.append(nn.Conv2d(int(out_channel / 4), int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), out_channel, (1, 1), bias=False))
        return res_block
       
        
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        conv1_x = self.proj2(x)
        x = self.complex_conv2(conv1_x)
        conv2_x = self.proj2(x)
#         x = self.complex_res1(conv1_x, conv2_x)
        res_x = self.proj2(conv2_x)
        x = self.complex_conv3(res_x)
        conv3_x = self.proj2(x)
#         x = self.complex_res2(res_x, conv3_x)
        x, losses = self.SURE(conv3_x, labels)
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
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2 + 0.05)
        return x, res_loss
    
    def clear_weights(self):
        self.SURE.clear_LE()
        return 0
    
class ManifoldNetRes1(nn.Module):
    def __init__(self, num_classes, num_distr, num_repeat):
        super(ManifoldNetRes1, self).__init__()
        params={'num_classes': num_classes, 'num_distr': num_distr, 'num_repeat': num_repeat}
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
        self.complex_conv3 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (1, 1))
        self.SURE = SURE_pure(params, calc_next(22, 5, 1, 20), 20)
        
        self.complex_res1 = complex.ResidualLayer(20, 20, 20, (5, 5), (2, 2))
        self.complex_res2 = complex.ResidualLayer(20, 20, 20, (5, 5), (1, 1))
        
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
        self.relu = nn.ReLU()
        
        self.conv_1 = nn.Conv2d(20, 30, (5,5))
        self.mp_1 = nn.MaxPool2d((3,3))
        self.conv_2 = nn.Conv2d(40, 50, (3,3))
        self.conv_3 = nn.Conv2d(60, 70, (2,2))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(50)
        self.bn_3 = nn.BatchNorm2d(70)
        self.linear_2 = nn.Linear(70, 40)
        self.linear_4 = nn.Linear(40, 11)
        self.name = 'Residual complex for mstar'
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        
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
        res_block.append(nn.Conv2d(int(out_channel / 4), int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), out_channel, (1, 1), bias=False))
        return res_block
       
        
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        conv1_x = self.proj2(x)
        x = self.complex_conv2(conv1_x)
        conv2_x = self.proj2(x)
#         x = self.complex_res1(conv1_x, conv2_x)
        res_x = self.proj2(conv2_x)
        x = self.complex_conv3(res_x)
        conv3_x = self.proj2(x)
#         x = self.complex_res2(res_x, conv3_x)
        
        x, losses = self.SURE(conv3_x, labels)
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
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss
    
    def clear_weights(self):
        self.SURE.clear_LE()
        return 0
