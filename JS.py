import torch 
import time, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions.log_normal as ln 
from pdb import set_trace as st
from torch.autograd import Variable
import torch.nn.functional as F

eps = 0.000001

def weightNormalize(weights, drop_prob=0.0):
    out = []
    for row in weights:
        if drop_prob==0.0:
            out.append(row**2/torch.sum(row**2))
        else:
            p = torch.randint(0, 2, (row.size())).float().cuda() 
            out.append((row**2/torch.sum(row**2))*p)
    return torch.stack(out)

def weightNormalize1(weights):
    return ((weights**2)/torch.sum(weights**2))


def weightNormalize2(weights):
    return weights/torch.sum(weights**2)



class SURE(nn.Module):
#     def __init__(self, num_classes, num_distr, input_shape):
    def __init__(self, params, in_channels, out_channels, kern_size, stride, input_shape):
        #input_shape must be torch.Size object
        super(SURE, self).__init__()
        num_classes = params['num_classes'] 
        num_distr = params['num_distr']
        self.num_repeat = 1
        input_shapes = list(input_shape)
        input_shapes[1] = out_channels
        self.out = out_channels
        
        shapes = torch.Size([num_classes]+input_shapes)
        self.classes = num_classes
        self.shapes = shapes
        self.num_distr = num_distr
        
        self.sigmas = Variable(torch.ones(self.classes), requires_grad=False).cuda()
        
        self.X_LEs = Variable(torch.zeros(shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((num_classes, 1))+eps, requires_grad=False).cuda() 
        self.w1 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        
        
        self.miu = torch.nn.Parameter(torch.rand(torch.Size([num_distr]+input_shapes)), requires_grad=True) #backprop
        self.tao = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True) #gets updated by backprop
        self.wFM = ComplexConv2Deffgroup(in_channels, out_channels, kern_size, stride)
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        
        self.ls = torch.nn.LogSigmoid()
        self.pool = torch.nn.MaxPool1d(kernel_size=num_classes, stride=num_classes)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))
        
    def forward(self, x_LE, labels=None, sigmas=None): 
        #Size of x is [B, features, in, H, W]
        #Size of sigmas is [num_distr]
        sigmas = self.sigmas ** 2
            
        w1 = weightNormalize1(self.w1)
        w2 = weightNormalize1(self.w1)
        #Apply wFM to inputs
        #x_LE is of shape [B, features, in, H, W]
        x_LE = self.wFM(x_LE)
        
        B, features, in_channel, H, W = x_LE.shape
        
        if labels is not None:
            #During training stage only:
            #Select Tensors of Same class and group together
            inputs = x_LE.view(B, -1)
            label_used = labels.unsqueeze(-1).repeat(1, torch.cumprod(torch.tensor(self.shapes[1:]), 0)[-1])

            temp_bins = self.X_LEs.view(self.classes, -1)
        
            self.X_LEs = temp_bins.scatter_add(0, label_used, inputs).reshape(self.shapes).detach()
            
            #Since we are taking the average, better keep track of number of samples in each class
            labels_weights = labels.unsqueeze(-1)
            src = torch.ones((labels.shape[0], 1))
            
            self.X_weights = self.X_weights.scatter_add(0, labels_weights, src.cuda()).detach()
            
        #Size of [num_classes, features, in, H, W]
        x_LE_out = (self.X_LEs.view(self.classes, -1) / self.X_weights).view(self.shapes)


        #Size of [num_classes, num_distr, features, in , H, W]
        x_LE_expand = x_LE_out.unsqueeze(1).repeat(1, self.num_distr, 1, 1, 1, 1)

        #Size of [num_distr, num_classes] 
        tao_sqrd = (self.tao ** 2).unsqueeze(-1).repeat(1, self.classes)

        #Size of [num_classes]
        sigma_sqrd = sigmas ** 2
        
        if labels is not None:
          #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###
            #Size of [num_distr, num_classes]
            term1 = sigma_sqrd / (tao_sqrd + sigma_sqrd) ** 2

            #Size of [num_distr, num_classes]
            LE_miu_dist = (self.ls(x_LE_expand).cuda() - self.ls(self.miu)) ** 2

            # Old way (original formula with log, replaced with sigmoidLog
    #             LE_miu_dist = (torch.log((x_LE_expand).cuda()+eps) - torch.log(self.miu+eps)) ** 2


            LE_miu_norm = torch.sum(LE_miu_dist.view(self.classes, self.num_distr, -1), dim=2).transpose(1, 0)

            #Size of [num_distr, num_classes]
            term2 = sigma_sqrd.unsqueeze(0).repeat(self.num_distr, 1) * LE_miu_norm

            #Size of [num_distr, num_classes]
            term3 = 2*self.out*H*W*(tao_sqrd ** 2 - sigma_sqrd ** 2).cuda() / self.X_weights.repeat(1, self.num_distr).transpose(1, 0).cuda()

            #Size of [num_distr]
            loss = torch.mean(term1 * (term2 + term3), dim=1)
        else:
            loss = None


      #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###


        #Size of [num_classes, num_distr] 
        tao_sqrd = (self.tao ** 2).unsqueeze(0).repeat(self.classes, 1)

        #Size of [num_classes, num_distr]
        sigma_sqrd = (sigmas ** 2).unsqueeze(-1).repeat(1, self.num_distr)

      #####THIS BLOCK OF CODE GENERATES THE MEANS FOR EACH CLASS###

        #These are of shape [num_classes, num_distr, in, H, W]
        theta_x_LE = x_LE_expand[:, :, 0, ...]
        mag_x_LE = x_LE_expand[:, :, 1, ...]

        x_LE_mag = (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) * self.ls(mag_x_LE+eps)

        miu_bins_mag = (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) *  self.ls(self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 1, ...]+eps)


        exp_sum_mag = torch.exp((x_LE_mag + miu_bins_mag) * w1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W))

        means_mag = torch.sum(exp_sum_mag, dim=1)

        x_LE_theta = mag_x_LE * (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        miu_bins_theta = self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 0, ...] * (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        means_theta = torch.sum((x_LE_theta + miu_bins_theta) * w2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W), dim=1)

        #[num_classes, features, in, H, W]
        means = torch.cat((means_theta.unsqueeze(1), means_mag.unsqueeze(1)), 1)

        means_expand = means.unsqueeze(1).repeat(1, B, 1, 1, 1, 1)

        x_LE = x_LE.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)

        dist_rot = self.SOmetric(x_LE[:, :, 0, ...].contiguous().view(-1), means_expand[:, :, 0, ...].contiguous().view(-1))

        dist_rot = dist_rot.view(self.classes, B, in_channel, H, W)   

        dist_abs = self.P1metric(x_LE[:, :, 1, ...].contiguous().view(-1), means_expand[:, :, 1, ...].contiguous().view(-1)).view(self.classes, B, in_channel, H, W)   

        #[num_classes, B, in, H, W]
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
#             dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        classes, B, in_channel, H, W = dist_l1.shape

        x_LE = dist_l1.permute(1, 2, 3, 4, 0).view(B, in_channel*H*W, self.classes) * (-1)

        x_LE = self.pool(x_LE).view(B, in_channel, H, W) * (-1)

        return x_LE, loss
        
    
    def clear_LE(self):
        self.X_LEs = Variable(torch.zeros(self.shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((self.classes, 1))+eps, requires_grad=False).cuda()
        
        
class SURE_pure(nn.Module):
#     def __init__(self, num_classes, num_distr, input_shape):
    def __init__(self, params, input_shape, out_channels):
        #input_shape must be torch.Size object
        super(SURE_pure, self).__init__()
        num_classes = params['num_classes'] 
        num_distr = params['num_distr']
        self.num_repeat = 1
        input_shapes = list(input_shape)
        input_shapes[1] = out_channels
        self.out = out_channels
        
        shapes = torch.Size([num_classes]+input_shapes)
        self.classes = num_classes
        self.shapes = shapes
        self.num_distr = num_distr
        
        self.sigmas = Variable(torch.ones(self.classes), requires_grad=False).cuda()
        
        self.X_LEs = Variable(torch.zeros(shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((num_classes, 1))+eps, requires_grad=False).cuda() 
        self.w1 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        
        
        self.miu = torch.nn.Parameter(torch.rand(torch.Size([num_distr]+input_shapes)), requires_grad=True) #backprop
        self.tao = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True) #gets updated by backprop
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        
        self.ls = torch.nn.LogSigmoid()
        self.pool = torch.nn.MaxPool1d(kernel_size=num_classes, stride=num_classes)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))
        
    def forward(self, x_LE, labels=None, sigmas=None): 
        #Size of x is [B, features, in, H, W]
        #Size of sigmas is [num_distr]
        sigmas = self.sigmas
            
        w1 = weightNormalize1(self.w1)
        w2 = weightNormalize1(self.w1)
        #Apply wFM to inputs
        #x_LE is of shape [B, features, in, H, W]
        B, features, in_channel, H, W = x_LE.shape
        
        if labels is not None:
            #During training stage only:
            #Select Tensors of Same class and group together
            inputs = x_LE.contiguous().view(B, -1)
            label_used = labels.unsqueeze(-1).repeat(1, torch.cumprod(torch.tensor(self.shapes[1:]), 0)[-1])

            temp_bins = self.X_LEs.view(self.classes, -1)
        
            self.X_LEs = temp_bins.scatter_add(0, label_used, inputs).reshape(self.shapes).detach()
            
            #Since we are taking the average, better keep track of number of samples in each class
            labels_weights = labels.unsqueeze(-1)
            src = torch.ones((labels.shape[0], 1))
            
            self.X_weights = self.X_weights.scatter_add(0, labels_weights, src.cuda()).detach()
            
        #Size of [num_classes, features, in, H, W]
        x_LE_out = (self.X_LEs.view(self.classes, -1) / self.X_weights).view(self.shapes)


        #Size of [num_classes, num_distr, features, in , H, W]
        x_LE_expand = x_LE_out.unsqueeze(1).repeat(1, self.num_distr, 1, 1, 1, 1)

        #Size of [num_distr, num_classes] 
        tao_sqrd = (self.tao ** 2).unsqueeze(-1).repeat(1, self.classes)

        #Size of [num_classes]
        sigma_sqrd = sigmas ** 2
        
        if labels is not None:
          #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###
            #Size of [num_distr, num_classes]
            term1 = sigma_sqrd / (tao_sqrd + sigma_sqrd) ** 2

            #Size of [num_distr, num_classes]
            LE_miu_dist = (self.ls(x_LE_expand).cuda() - self.ls(self.miu)) ** 2

            # Old way (original formula with log, replaced with sigmoidLog
    #             LE_miu_dist = (torch.log((x_LE_expand).cuda()+eps) - torch.log(self.miu+eps)) ** 2


            LE_miu_norm = torch.sum(LE_miu_dist.view(self.classes, self.num_distr, -1), dim=2).transpose(1, 0)

            #Size of [num_distr, num_classes]
            term2 = sigma_sqrd.unsqueeze(0).repeat(self.num_distr, 1) * LE_miu_norm

            #Size of [num_distr, num_classes]
            term3 = 2*self.out*H*W*(tao_sqrd ** 2 - sigma_sqrd ** 2).cuda() / self.X_weights.repeat(1, self.num_distr).transpose(1, 0).cuda()

            #Size of [num_distr]
            loss = torch.mean(term1 * (term2 + term3), dim=1)
        else:
            loss = None


      #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###


        #Size of [num_classes, num_distr] 
        tao_sqrd = (self.tao ** 2).unsqueeze(0).repeat(self.classes, 1)

        #Size of [num_classes, num_distr]
        sigma_sqrd = (sigmas ** 2).unsqueeze(-1).repeat(1, self.num_distr)

      #####THIS BLOCK OF CODE GENERATES THE MEANS FOR EACH CLASS###

        #These are of shape [num_classes, num_distr, in, H, W]
        theta_x_LE = x_LE_expand[:, :, 0, ...]
        mag_x_LE = x_LE_expand[:, :, 1, ...]

        x_LE_mag = (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) * self.ls(mag_x_LE+eps)

        miu_bins_mag = (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) *  self.ls(self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 1, ...]+eps)


        exp_sum_mag = torch.exp((x_LE_mag + miu_bins_mag) * w1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W))

        means_mag = torch.sum(exp_sum_mag, dim=1)

        x_LE_theta = mag_x_LE * (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        miu_bins_theta = self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 0, ...] * (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        means_theta = torch.sum((x_LE_theta + miu_bins_theta) * w2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W), dim=1)

        #[num_classes, features, in, H, W]
        means = torch.cat((means_theta.unsqueeze(1), means_mag.unsqueeze(1)), 1)

        means_expand = means.unsqueeze(1).repeat(1, B, 1, 1, 1, 1)

        x_LE = x_LE.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)

        dist_rot = self.SOmetric(x_LE[:, :, 0, ...].contiguous().view(-1), means_expand[:, :, 0, ...].contiguous().view(-1))

        dist_rot = dist_rot.view(self.classes, B, in_channel, H, W)   

        dist_abs = self.P1metric(x_LE[:, :, 1, ...].contiguous().view(-1), means_expand[:, :, 1, ...].contiguous().view(-1)).view(self.classes, B, in_channel, H, W)   

        #[num_classes, B, in, H, W]
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
#             dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        classes, B, in_channel, H, W = dist_l1.shape

        x_LE = dist_l1.permute(1, 2, 3, 4, 0).view(B, in_channel*H*W, self.classes) * (-1)

        x_LE = self.pool(x_LE).view(B, in_channel, H, W) * (-1)

        return x_LE, loss
        
    
    def clear_LE(self):
        self.X_LEs = Variable(torch.zeros(self.shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((self.classes, 1))+eps, requires_grad=False).cuda()
        

    

class SURE_pure4D(nn.Module):
#     def __init__(self, num_classes, num_distr, input_shape):
    def __init__(self, params, input_shape, out_channels):
        #input_shape must be torch.Size object
        super(SURE_pure4D, self).__init__()
        num_classes = params['num_classes'] 
        num_distr = params['num_distr']
        self.num_repeat = 1
        input_shapes = list(input_shape)
        input_shapes[1] = out_channels
#         input_shapes[0] = 4 #Hardcode 4D
        self.out = out_channels
        
        shapes = torch.Size([num_classes]+input_shapes)
        self.classes = num_classes
        self.shapes = shapes
        self.num_distr = num_distr
        
        self.sigmas = Variable(torch.ones(self.classes), requires_grad=False).cuda()
        
        self.X_LEs = Variable(torch.zeros(shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((num_classes, 1))+eps, requires_grad=False).cuda() 
        
        self.X_LEs_xy = Variable(torch.zeros(shapes)+eps, requires_grad=False).cuda()
        
        self.w1 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        
        
        self.miu = torch.nn.Parameter(torch.rand(torch.Size([num_distr]+input_shapes)), requires_grad=True) #backprop
        self.tao = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True) #gets updated by backprop
        self.weight = torch.nn.Parameter(torch.rand([3]), requires_grad=True)
        
        self.ls = torch.nn.LogSigmoid()
        self.pool = torch.nn.MaxPool1d(kernel_size=num_classes, stride=num_classes)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
    def Euclmetric(self, X, Y):
        return torch.sqrt(torch.sum(X-Y, dim=1))
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))
        
    def forward(self, x_LE, labels=None, sigmas=None): 
        #Size of x is [B, features, in, H, W]
        #Size of sigmas is [num_distr]
       
        sigmas = self.sigmas
            
        w1 = weightNormalize1(self.w1)
        w2 = weightNormalize1(self.w1)
        #Apply wFM to inputs
        #x_LE is of shape [B, features, in, H, W]
        B, features, in_channel, H, W = x_LE.shape
        
        x_LE_xy = x_LE[:, 2:, ...]
        x_LE = x_LE[:, :2, ...]
        if labels is not None:
            #During training stage only:
            #Select Tensors of Same class and group together
            inputs = x_LE.contiguous().view(B, -1)
            inputs_xy = x_LE_xy.contiguous().view(B, -1)
            
            label_used = labels.unsqueeze(-1).repeat(1, torch.cumprod(torch.tensor(self.shapes[1:]), 0)[-1])

            temp_bins = self.X_LEs.view(self.classes, -1)
            temp_bins_xy = self.X_LEs_xy.view(self.classes, -1)
            
            self.X_LEs = temp_bins.scatter_add(0, label_used, inputs).reshape(self.shapes).detach()
            self.X_LEs_xy = temp_bins_xy.scatter_add(0, label_used, inputs_xy).reshape(self.shapes).detach()
            #Since we are taking the average, better keep track of number of samples in each class
            labels_weights = labels.unsqueeze(-1)
            src = torch.ones((labels.shape[0], 1))
            
            self.X_weights = self.X_weights.scatter_add(0, labels_weights, src.cuda()).detach()
            
        #Size of [num_classes, features, in, H, W]
        x_LE_out = (self.X_LEs.view(self.classes, -1) / self.X_weights).view(self.shapes)


        #Size of [num_classes, num_distr, features, in , H, W]
        x_LE_expand = x_LE_out.unsqueeze(1).repeat(1, self.num_distr, 1, 1, 1, 1)

        #Size of [num_distr, num_classes] 
        tao_sqrd = (self.tao ** 2).unsqueeze(-1).repeat(1, self.classes)

        #Size of [num_classes]
        sigma_sqrd = sigmas ** 2
        
        if labels is not None:
          #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###
            #Size of [num_distr, num_classes]
            term1 = sigma_sqrd / (tao_sqrd + sigma_sqrd) ** 2

            #Size of [num_distr, num_classes]
            LE_miu_dist = (self.ls(x_LE_expand).cuda() - self.ls(self.miu)) ** 2

            # Old way (original formula with log, replaced with sigmoidLog
    #             LE_miu_dist = (torch.log((x_LE_expand).cuda()+eps) - torch.log(self.miu+eps)) ** 2


            LE_miu_norm = torch.sum(LE_miu_dist.view(self.classes, self.num_distr, -1), dim=2).transpose(1, 0)

            #Size of [num_distr, num_classes]
            term2 = sigma_sqrd.unsqueeze(0).repeat(self.num_distr, 1) * LE_miu_norm

            #Size of [num_distr, num_classes]
            term3 = 2*self.out*H*W*(tao_sqrd ** 2 - sigma_sqrd ** 2).cuda() / self.X_weights.repeat(1, self.num_distr).transpose(1, 0).cuda()

            #Size of [num_distr]
            loss = torch.mean(term1 * (term2 + term3), dim=1)
        else:
            loss = None


      #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###


        #Size of [num_classes, num_distr] 
        tao_sqrd = (self.tao ** 2).unsqueeze(0).repeat(self.classes, 1)

        #Size of [num_classes, num_distr]
        sigma_sqrd = (sigmas ** 2).unsqueeze(-1).repeat(1, self.num_distr)

      #####THIS BLOCK OF CODE GENERATES THE MEANS FOR EACH CLASS###

        #These are of shape [num_classes, num_distr, in, H, W]
        theta_x_LE = x_LE_expand[:, :, 0, ...]
        mag_x_LE = x_LE_expand[:, :, 1, ...]
        x_LE_mag = (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) * self.ls(mag_x_LE+eps)

        miu_bins_mag = (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) *  self.ls(self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 1, ...]+eps)


        exp_sum_mag = torch.exp((x_LE_mag + miu_bins_mag) * w1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W))

        means_mag = torch.sum(exp_sum_mag, dim=1)

        x_LE_theta = mag_x_LE * (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        miu_bins_theta = self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 0, ...] * (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        means_theta = torch.sum((x_LE_theta + miu_bins_theta) * w2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W), dim=1)

        #[num_classes, features, in, H, W]
        means = torch.cat((means_theta.unsqueeze(1), means_mag.unsqueeze(1)), 1)

        means_expand = means.unsqueeze(1).repeat(1, B, 1, 1, 1, 1)

        x_LE = x_LE.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)

        dist_rot = self.SOmetric(x_LE[:, :, 0, ...].contiguous().view(-1), means_expand[:, :, 0, ...].contiguous().view(-1))

        dist_rot = dist_rot.view(self.classes, B, in_channel, H, W)   

        dist_abs = self.P1metric(x_LE[:, :, 1, ...].contiguous().view(-1), means_expand[:, :, 1, ...].contiguous().view(-1)).view(self.classes, B, in_channel, H, W)   
        
        (self.X_LEs.view(self.classes, -1) / self.X_weights).view(self.shapes)
        
        #[classes, 2, in, H, W]
        x_LEs_xy_out = (self.X_LEs_xy.view(self.classes, -1) / self.X_weights).view(self.shapes)
        x_LEs_xy_out = x_LEs_xy_out.unsqueeze(1).repeat(1, B, 1, 1, 1, 1)
        x_LE_xy = x_LE_xy.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)
        
        #[class, B, 2, in, H, W]
        dist_xy = (x_LE_xy - x_LEs_xy_out) ** 2
        dist_xy = torch.sum(dist_xy, dim=2)

        #[num_classes, B, in, H, W]
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_xy
#             dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        classes, B, in_channel, H, W = dist_l1.shape

        x_LE = dist_l1.permute(1, 2, 3, 4, 0).view(B, in_channel*H*W, self.classes) * (-1)

        x_LE = self.pool(x_LE).view(B, in_channel, H, W) * (-1)
        
        

        return x_LE, loss
        
    
    def clear_LE(self):
        self.X_LEs = Variable(torch.zeros(self.shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((self.classes, 1))+eps, requires_grad=False).cuda()






















def weightNormalize1(weights):
    return ((weights**2)/torch.sum(weights**2))


def weightNormalize2(weights):
    return weights/torch.sum(weights**2)


class ComplexConv2Deffgroup(nn.Module):
    
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(ComplexConv2Deffgroup, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.wmr = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.wma = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True) 
        self.complex_conv = ComplexConv2Deffangle(in_channels, out_channels, kern_size, stride)

    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        temporal_buckets_rot = temporal_buckets[:,0,...]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        
        tbr_shape0 = temporal_buckets_rot.shape
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1], tbr_shape0[2])
        temporal_buckets_abs = temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1],tbr_shape0[2])
        tbr_shape = temporal_buckets_rot.shape 
        
        in_rot = temporal_buckets_rot * weightNormalize2(self.wmr)
        in_abs = temporal_buckets_abs + weightNormalize1(self.wma)
        in_rot = in_rot.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
        in_abs = in_abs.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
        in_ = torch.cat((in_rot, in_abs), 1).view(tbr_shape0[0], -1, out_spatial_x*out_spatial_y)
        in_fold = nn.Fold(output_size=(x_shape[3],x_shape[4]), kernel_size=self.kern_size, stride=self.stride)(in_)
        in_fold = in_fold.view(x_shape[0],x_shape[1],x_shape[2],x_shape[3],x_shape[4])
        out = self.complex_conv(in_fold)
        
        return out 
    
class ComplexConv2Deffangle(nn.Module):
    
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
    
    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        temporal_buckets_rot = temporal_buckets[:,0,...]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        tbr_shape = temporal_buckets_rot.shape 
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        tba_shape = temporal_buckets_abs.shape   
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        return torch.cat((out_rot,out_abs),1)


    
#     def create_M(self, miu, lbd):
#         distr = ln(miu, lbd*np.eye(miu.shape))
#         return distr


#Block of code for EM algorithm
#             #Calculate the R matrix of shape [num_classes, num_distr]
#             #This utilizes self.w, which is of shape [num_distr]
            
#             #Shape [num_classes, num_distr, features, in, H, W]
#             means_dup = means.unsqueeze(1).repeat(1, self.num_distr, 1, 1, 1, 1)
            
#             #Shape [num_classes, num_distr, features, in, H, W]
#             mius = self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)
            
#             #(x-miu) step
#             means_normalized = means_dup - mius
            
#             #Shape [num_classes, num_distr, 1, features*in*H*W]
#             means_stretched = means_normalized.view(self.classes*self.num_distr, 1, -1)
            
            
#             transposed = means_stretched.transpose(2, 1)
            
#             R = torch.bmm(means_stretched, transposed).squeeze(-1).squeeze(-1).view(self.classes, self.num_distr).transpose(1, 0) / sigmas
#             R = R.transpose(1, 0)
            
            
#             if True in torch.isnan(self.miu).cpu().detach().numpy():
#                 print("MIU!")
#                 st()
                
#             if True in torch.isnan(self.tao).cpu().detach().numpy():
#                 print("TAO!")
#                 st()
                
                
#             #R is of shape [num_classes, num_distr]
#             R_ = ( (2*math.pi) ** self.num_distr * torch.cumprod(sigmas, dim=0)[-1]) ** (-0.5) * torch.exp(-0.5 * R + eps)
            
#             R_ = R_ * w.unsqueeze(0).repeat(self.classes, 1) + eps
            
#             sum_R = torch.sum(R_, dim=1, keepdim=True)
            
#             R = R_ / sum_R
            
            
#             #Update ws!:
#             self.w = torch.mean(R, dim=0).detach()




#         #NEED CHANGE
#             #Size of [num_classes, num_distr, features, in, H, W]
#             x_LE_expand = (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features, in_channel, H, W) * self.ls(x_LE_expand)
            
#             #Size of [num_classes, num_distr, features, in, H, W]
#             miu_bins = (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features, in_channel, H, W) * self.ls(self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1))

            
            
# #             Old way (original formula with log, replaced with sigmoidLog
# #              x_LE_expand = (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features, in_channel, H, W) * torch.log(F.relu(x_LE_expand)+eps)

# #             miu_bins = (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, features, in_channel, H, W) * torch.log(F.relu(self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1))+eps)

 
#             #Size of [num_classes, num_distr, features, in, H, W]
#             exp_sum = torch.exp(x_LE_expand + miu_bins)

#             #Size of [num_classes, features, in, H, W]: This is the Mis that we need
#             means = torch.sum(exp_sum * w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, features, in_channel, H, W), dim=1)
          #####THIS BLOCK OF CODE GENERATES THE MEANS FOR EACH CLASS###

    
