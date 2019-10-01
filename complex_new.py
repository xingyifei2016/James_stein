import torch 
import time, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from pdb import set_trace as st


#Y = torch.tensor([[0, self.weight_rot],[-self.weight_rot, 0]])
#os.environ["CUDA_VISIBLE_DEVICES"] ='0,2' 
#import fm_ops as spd_ops
eps = 0.000001

#device = torch.device('cpu')

def b_inv(b_mat):
    #b_mat = b_mat.cpu()
    #eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    #b_inv, _ = torch.gesv(eye, b_mat)
    #b_inv = b_inv.to(device)
    #print(b_inv.contiguous())
    #b = [t.inverse() for t in torch.unbind(b_mat)]
    #b_inv = torch.stack(b)
    b00 = b_mat[:,0,0]
    b01 = b_mat[:,0,1]
    b10 = b_mat[:,1,0]
    b11 = b_mat[:,1,1]
    det = (b00*b11-b01*b10)
    b00 = b00/ (det+eps)
    b01 = b01/ (det+eps)
    b10 = b10/ (det+eps)
    b11 = b11/ (det+eps)
    b_inv1 = torch.cat((torch.cat((b11.view(-1,1,1),-1.*b01.view(-1,1,1)),dim=2),torch.cat((-1.*b10.view(-1,1,1),b00.view(-1,1,1)),dim=2)),dim=1)
    return b_inv1


def weightNormalizexy(weights1, weights2):
#     out = []
#     nw = torch.norm(torch.cat((weights1.unsqueeze(0), weights2.unsqueeze(0)),0), dim=0)
#     for row in nw:
#         out.append(row/torch.sum(row))
    weights = (weights1+weights2)**2
    weights = weights / torch.sum(weights)
    return weights

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


class ReLU4Dspnlog(nn.Module):
    def __init__(self,channels):
        super(ReLU4Dspnlog, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        #self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels
        self.relu = nn.ReLU() 
        #self.Y = torch.nn.Parameter(torch.tensor([[torch.cos(self.weight_rot**2), torch.sin(self.weight_rot**2)], [-torch.sin(self.weight_rot**2), torch.cos(self.weight_rot**2)]]), requires_grad=True)
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0,...]
        temp_abs = x[:,1,...]  
        temp_x = self.relu(x[:,2,...]).unsqueeze(1)
        temp_y = self.relu(x[:,3,...]).unsqueeze(1)
        temp_rot_prod = (temp_rot.unsqueeze(1)*(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        #print(temp_rot_prod.shape)
        #print(temp_abs.shape)
        #print(self.weight_rot)
        return torch.cat((temp_rot_prod, temp_abs, temp_x, temp_y),1)
 





class ReLU4Dsp(nn.Module):
    def __init__(self,channels):
        super(ReLU4Dsp, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        #self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels
        self.relu = nn.ReLU() 
        #self.Y = torch.nn.Parameter(torch.tensor([[torch.cos(self.weight_rot**2), torch.sin(self.weight_rot**2)], [-torch.sin(self.weight_rot**2), torch.cos(self.weight_rot**2)]]), requires_grad=True)
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0,...]
        temp_abs = x[:,1,...]  
        temp_x = self.relu(x[:,2,...]).unsqueeze(1)
        temp_y = self.relu(x[:,3,...]).unsqueeze(1)
        temp_rot_prod = (temp_rot.unsqueeze(1)*(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize1(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        #print(temp_rot_prod.shape)
        #print(temp_abs.shape)
        #print(self.weight_rot)
        return torch.cat((temp_rot_prod, temp_abs, temp_x, temp_y),1)
 





class ReLU4D(nn.Module):
    def __init__(self,channels):
        super(ReLU4D, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels
        self.relu = nn.ReLU() 
        #self.Y = torch.nn.Parameter(torch.tensor([[torch.cos(self.weight_rot**2), torch.sin(self.weight_rot**2)], [-torch.sin(self.weight_rot**2), torch.cos(self.weight_rot**2)]]), requires_grad=True)
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0,...]
        temp_abs = x[:,1,...]  
        temp_x = self.relu(x[:,2,...]).unsqueeze(1)
        temp_y = self.relu(x[:,3,...]).unsqueeze(1)
        temp_rot_prod = (temp_rot.unsqueeze(1)*(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize1(self.weight_abs+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        #print(temp_rot_prod.shape)
        #print(temp_abs.shape)
        #print(self.weight_rot)
        return torch.cat((temp_rot_prod, temp_abs, temp_x, temp_y),1)
 



class euclidReLU(nn.Module):
    def __init__(self):
        super(euclidReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        y = torch.zeros((x.shape)).cuda()
        y[:,0,...] = x[:,1,...]*torch.cos(x[:,0,...])
        y[:,1,...] = x[:,1,...]*torch.sin(x[:,0,...])
        y = self.relu(y)+eps
        M = torch.norm(y, dim=1)
        T = torch.atan2(y[:,0,...], y[:,1,...])
        return torch.cat((T.unsqueeze(1), M.unsqueeze(1)),1)
 







class manifoldReLUv2angle(nn.Module):
    def __init__(self,channels):
        super(manifoldReLUv2angle, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels 
        #self.Y = torch.nn.Parameter(torch.tensor([[torch.cos(self.weight_rot**2), torch.sin(self.weight_rot**2)], [-torch.sin(self.weight_rot**2), torch.cos(self.weight_rot**2)]]), requires_grad=True)
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0,...]
        temp_abs = x[:,1,...]  
        temp_rot_prod = (temp_rot.unsqueeze(1)*(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize1(self.weight_abs+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        #print(temp_rot_prod.shape)
        #print(temp_abs.shape)
        #print(self.weight_rot)
        return torch.cat((temp_rot_prod, temp_abs),1)
 





class manifoldReLUv2(nn.Module):
    def __init__(self, channels):
        super(manifoldReLUv2, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,1,channels), requires_grad=True)
        self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels 
        #self.Y = torch.nn.Parameter(torch.tensor([[torch.cos(self.weight_rot**2), torch.sin(self.weight_rot**2)], [-torch.sin(self.weight_rot**2), torch.cos(self.weight_rot**2)]]), requires_grad=True)
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0:4,...].view(x_shape[0],2,2,x_shape[2],x_shape[3],-1)
        temp_abs = x[:,4,...]  
        zero = torch.zeros(1,1,self.channels).cuda()
        Y = torch.cat((torch.cat((zero,self.weight_rot),1), torch.cat((-self.weight_rot,zero),1)),0).permute(2,0,1).contiguous()
        #Y = torch.tensor([[0, self.weight_rot],[-self.weight_rot, 0]])
        
        eye = Y.new_ones(Y.size(-1)).diag().expand_as(Y)
        #I = torch.eye(2)
        YY = torch.bmm((eye-Y), b_inv(eye+Y))
        temp_rot = temp_rot.permute(0,4,5,3,1,2).contiguous().view(-1,2,2)
        temp_rot_prod = torch.bmm(temp_rot, YY.unsqueeze(0).repeat(x_shape[0]*x_shape[3]*x_shape[4],1,1,1).view(-1,2,2))
        temp_rot_prod = temp_rot_prod.view(x_shape[0],x_shape[3],x_shape[4],x_shape[2],2,2).permute(0,4,5,3,1,2).contiguous().view(-1,4,x_shape[2],x_shape[3],x_shape[4])
        temp_abs = (temp_abs.unsqueeze(1)*(self.weight_abs**2+eps).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        #print(temp_rot_prod.shape)
        #print(temp_abs.shape)
        #print(self.weight_rot)
        return torch.cat((temp_rot_prod, temp_abs),1)
 
class manifoldReLU(nn.Module):
    def __init__(self):
        super(manifoldReLU, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.weight_abs = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        
        #self.Y = torch.nn.Parameter(torch.tensor([[torch.cos(self.weight_rot**2), torch.sin(self.weight_rot**2)], [-torch.sin(self.weight_rot**2), torch.cos(self.weight_rot**2)]]), requires_grad=True)
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0:4,...].view(x_shape[0],2,2,x_shape[2],x_shape[3],-1)
        temp_abs = x[:,4,...]  
        
        Y = torch.tensor([[0, self.weight_rot],[-self.weight_rot, 0]])
        I = torch.eye(2)
        YY = torch.mm((I-Y), torch.inverse(I+Y)).cuda()
        temp_rot = temp_rot.permute(0,3,4,5,1,2).contiguous().view(-1,2,2)
        temp_rot_prod = torch.bmm(temp_rot, YY.unsqueeze(0).repeat(temp_rot.shape[0],1,1))
        temp_rot_prod = temp_rot_prod.view(x_shape[0],x_shape[2],x_shape[3],x_shape[4],2,2).permute(0,4,5,1,2,3).contiguous().view(-1,4,x_shape[2],x_shape[3],x_shape[4])
        temp_abs = (temp_abs*(self.weight_abs**2+eps)).unsqueeze(1)
        #print(temp_rot_prod.shape)
        #print(temp_abs.shape)
        #print(self.weight_rot)
        return torch.cat((temp_rot_prod, temp_abs),1)


class ComplexConv2DEuc(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(ComplexConv2DEuc, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.weight_matrix_rot = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = torch.acos(torch.clamp(temporal_buckets[:,0,...],-1.0,1.0))
        
        #Shape: [batches, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,4,...]
        
        tbr_shape = temporal_buckets_rot.shape
        
        temporal_buckets_rot = temporal_buckets_rot.unsqueeze(1).repeat(1,self.out_channels,1,1) 
        
        tba_shape = temporal_buckets_rot.shape
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1)
        
        out_rot_C = torch.cos(out_rot).unsqueeze(1)
        out_rot_S = torch.sin(out_rot).unsqueeze(1)
        
        #Shape: [batches, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps).unsqueeze(1).repeat(1,self.out_channels,1,1))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        out_abs = torch.exp((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).unsqueeze(1)
        out_abs_shape = out_abs.shape
         
        out_rot = (torch.cat((out_rot_C,out_rot_S,-out_rot_S,out_rot_C),1)).view(out_abs_shape[0],4, out_abs_shape[2], out_spatial_x, out_spatial_y)
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        out_abs = out_abs.view(out_abs_shape[0],out_abs_shape[1], out_abs_shape[2], out_spatial_x, out_spatial_y)
        #print(out_rot.shape,out_abs.shape)
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs),1)


class ComplexConv2Deffcomplex(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1 = torch.nn.Parameter(torch.rand(2,in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2 = torch.nn.Parameter(torch.rand(2,out_channels, in_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_transposed_bucket = x
        temporal_transposed_bucket[:,0,...] = x[:,1,...]
        temporal_transposed_bucket[:,1,...] = x[:,0,...]
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        tbr_shape = temporal_transposed_bucket.shape 
        
        #shape: [batches*L, features, in_channels]
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,4,1,2,3).contiguous().view(-1, tbr_shape[1], tbr_shape[2], tbr_shape[3])*(self.weight_matrix_rot1),3)))
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.view(-1,tbr_shape[1],1,self.in_channels).repeat(1,1,self.out_channels,1)
        
        
        #shape: [batches, out
        out_rot = (torch.sum(out_rot*(self.weight_matrix_rot2),3)).view(tbr_shape[0], out_spatial_x, out_spatial_y, tbr_shape[1],self.out_channels).permute(0,3,4,1,2).contiguous()
 
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs),1)


class ComplexBNormlearnable(nn.Module):
      def __init__(self, num_channels):
          super(ComplexBNormlearnable, self).__init__()
          self.num_channels = num_channels
          self.mean = torch.nn.Parameter(torch.rand(2, num_channels), requires_grad=True)
          self.std = torch.nn.Parameter(torch.rand(2,num_channels), requires_grad=True)
      def forward(self, x):
          #shape: [batches, 2, channels, spatial_x, spatial_y]
          x_shape = x.shape
          y = torch.zeros((x.shape)).cuda()
          y[:,0,...] = x[:,1,...]*torch.cos(x[:,0,...])
          y[:,1,...] = x[:,1,...]*torch.sin(x[:,0,...])
          y = y.permute(0,3,4,1,2).contiguous().view(-1,x_shape[1],x_shape[2])
          z = ((y - weightNormalize2(self.mean))/(weightNormalize1(self.std)+eps)).view(x_shape[0],x_shape[3],x_shape[4],x_shape[1],x_shape[2]).permute(0,3,4,1,2).contiguous()
          M = torch.norm(z, dim=1)
          T = torch.atan2(z[:,1,...], z[:,0,...])
          return torch.cat((T.unsqueeze(1), M.unsqueeze(1)),1)


class ComplexBNormpolar(nn.Module):
      def __init__(self):
          super(ComplexBNormpolar, self).__init__()
      def forward(self, x):
          #shape: [batches, 2, channels, spatial_x, spatial_y]
          x_shape = x.shape
          y = torch.zeros((x.shape)).cuda()
          y[:,0,...] = x[:,0,...]
          y[:,1,...] = x[:,1,...]
          y = y.permute(0,3,4,1,2).contiguous().view(-1,x_shape[1],x_shape[2])
          mean = torch.zeros((2, x_shape[2])).cuda()
          std = torch.zeros((2, x_shape[2])).cuda()
          mean[0,:] = torch.mean(y[:,0,...], 0)
          std[0,:] = torch.std(y[:,0,...], 0)
          mean[1,:] = torch.min(y[:,1,...], dim=0)[0]
          std[1,:] = torch.max(y[:,1,...], dim=0)[0]
          z = ((y - mean+eps)/(std+eps)).view(x_shape[0],x_shape[3],x_shape[4],x_shape[1],x_shape[2]).permute(0,3,4,1,2).contiguous()
          M = torch.norm(z, dim=1)
          T = torch.atan2(z[:,1,...], z[:,0,...])
          return torch.cat((T.unsqueeze(1), M.unsqueeze(1)),1)


class ComplexBNormspatial(nn.Module):
      def __init__(self, bnorm):
          super(ComplexBNormspatial, self).__init__()
          self.bnorm = bnorm
      def forward(self, x):
          #b1 = nn.BatchNorm2d(self.num_channels)
          x_shape = x.shape
          #print(x_shape)
          return torch.cat((self.bnorm(x[:,0:2,...].contiguous().view(-1, 2*x_shape[2],x_shape[3],x_shape[4])).view(-1, 2,x_shape[2],x_shape[3],x_shape[4]), x[:,2:,...]),1)



class ComplexBNorm(nn.Module):
      def __init__(self):
          super(ComplexBNorm, self).__init__()
      def forward(self, x):
          #shape: [batches, 2, channels, spatial_x, spatial_y]
          x_shape = x.shape
          y = torch.zeros((x.shape)).cuda()
          y[:,0,...] = x[:,1,...]*torch.cos(x[:,0,...])
          y[:,1,...] = x[:,1,...]*torch.sin(x[:,0,...])
          y = y.permute(0,3,4,1,2).contiguous().view(-1,x_shape[1],x_shape[2])
          mean = torch.zeros((2, x_shape[2])).cuda()
          std = torch.zeros((2, x_shape[2])).cuda() 
          mean[0,:] = torch.mean(y[:,0,...], 0)
          std[0,:] = torch.std(y[:,0,...], 0)
          mean[1,:] = torch.mean(y[:,1,...], 0)
          std[1,:] = torch.std(y[:,1,...], 0)
          z = ((y - mean)/(std+eps)).view(x_shape[0],x_shape[3],x_shape[4],x_shape[1],x_shape[2]).permute(0,3,4,1,2).contiguous()
          M = torch.norm(z, dim=1)
          T = torch.atan2(z[:,1,...], z[:,0,...])
          return torch.cat((T.unsqueeze(1), M.unsqueeze(1)),1)


class ComplexRes(nn.Module):
      def __init__(self, channels):
          super(ComplexRes, self).__init__()
          self.channels = channels
          self.w = torch.nn.Parameter(torch.rand(channels), requires_grad=True)
      def forward(self, x, y):
          #shape: [batches, features, in_channels, spatial_x, spatial_y]
          x_shape = x.shape
          y_shape = y.shape
          m1 = torch.min([x_shape[3], y_shape[3]])
          m2 = torch.min([x_shape[4], y_shape[4]])
          z = torch.zeros((x_shape[0], x_shape[1], x_shape[2], m1, m2)).cuda()
          eye = torch.ones((channels)).cuda()
          z = weightNormalize1(self.w)*x[:,:,:,:m1,:m2].permute(0,1,3,4,2).contiguous().view(-1,self.channels) + (eye - weightNormalize1(self.w))*y[:,:,:,:m1,:m2].permute(0,1,3,4,2).contiguous().view(-1,self.channels)
          z = z.view(x_shape[0],x_shape[1],m1,m2,x_shape[2]).permute(0,1,4,2,3).contiguous()
          return z
          

class ResidualLayer(nn.Module):
    
    def __init__(self, in_channel1, in_channel2, out_channel, kern_size, stride):
        super(ResidualLayer, self).__init__()
        
        #We assume in_channel1 has larger spatial resolution
        
        self.in_channel1 = in_channel1
        self.in_channel2 = in_channel2
        self.out_channel = out_channel
        self.transform = ComplexConv2Deffgroup(in_channel1, in_channel1, kern_size, stride)
        self.w = torch.nn.Parameter(torch.rand(out_channel, in_channel1+in_channel2), requires_grad=True) 
        
    def forward(self, x1, x2):
        #Assume all square spatial resolutions, x1 has higher resolution
        x1 = self.transform(x1)
        x = torch.cat((x1, x2), 2)
        
        #B, in, H, W
        x_rot = x[:, 0, :, :, :]
        out_rot = x_rot.permute(0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, self.out_channel, 1)
        out_rot = torch.sum(out_rot*weightNormalize1(self.w),-1).unsqueeze(1)
        x_abs = x[:, 1, :, :, :]
        
        x_abs = x_abs.permute(0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, self.out_channel, 1)
        out_abs = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize1(self.w), -1)).unsqueeze(1)
       
        return torch.cat((out_rot, out_abs), 1).permute(0, 1, 4, 2, 3)
        
        
            
    

class ComplexConv2Deffgroup(nn.Module):
      def __init__(self, in_channels, out_channels, kern_size, stride, do_conv=True):
          super(ComplexConv2Deffgroup, self).__init__()
          self.in_channels = in_channels
          self.out_channels = out_channels
          self.kern_size = kern_size
          self.stride = stride
          self.do_conv = do_conv
          #self.wmr1 = torch.nn.Parameter(torch.rand(kern_size[0]*kern_size[1]), requires_grad=True)
          #self.wma1 = torch.nn.Parameter(torch.rand(kern_size[0]*kern_size[1]), requires_grad=True)
          #self.wmr2 = torch.nn.Parameter(torch.rand(in_channels), requires_grad=True)
          #self.wma2 = torch.nn.Parameter(torch.rand(in_channels), requires_grad=True)
          self.wmr = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
          self.wma = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True) 
          #self.wmr3 = torch.nn.Parameter(torch.rand(out_channels), requires_grad=True)
          #self.wma3 = torch.nn.Parameter(torch.rand(out_channels), requires_grad=True)
          #self.w1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
          #self.w2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True) 
          if do_conv: 
             self.complex_conv = ComplexConv2Deffangle(in_channels, out_channels, kern_size, stride)
          else:
             self.new_wr = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
             self.new_wa = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
      
      def ComplexweightedMean(self, x_rot, x_abs):
           x_shape = x_rot.shape
           out_rot = torch.sum(x_rot*weightNormalize1(self.w1), 2).unsqueeze(1).repeat(1, self.out_channels, 1)
           out_rot = torch.sum(out_rot*weightNormalize1(self.w2),2)
           x_abs_log = torch.log(x_abs+eps)
           out_abs = torch.sum(x_abs_log*weightNormalize1(self.w1), 2).unsqueeze(1).repeat(1, self.out_channels, 1)
           out_abs = torch.exp(torch.sum(out_abs*weightNormalize1(self.w2), 2))
           return (out_rot,out_abs)

      def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        
        tbr_shape0 = temporal_buckets_rot.shape
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1], tbr_shape0[2])
        temporal_buckets_abs = temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1],tbr_shape0[2])
        #shape: [batches*L, in_channels, kern_size[0]*kern_size[1]]
        tbr_shape = temporal_buckets_rot.shape 
        
        in_rot = temporal_buckets_rot * weightNormalize2(self.wmr)#(self.wmr1).unsqueeze(0).repeat(tbr_shape[1], 1) + (self.wmr2).unsqueeze(1).repeat(1, tbr_shape[2]))
        in_abs = temporal_buckets_abs + weightNormalize1(self.wma)#(self.wma1).unsqueeze(0).repeat(tbr_shape[1], 1) + (self.wma2).unsqueeze(1).repeat(1, tbr_shape[2]))
        #print(temporal_buckets_abs)
        if self.do_conv:
           in_rot = in_rot.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
           in_abs = in_abs.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
           in_ = torch.cat((in_rot, in_abs), 1).view(tbr_shape0[0], -1, out_spatial_x*out_spatial_y)
           in_fold = nn.Fold(output_size=(x_shape[3],x_shape[4]), kernel_size=self.kern_size, stride=self.stride)(in_)
           in_fold = in_fold.view(x_shape[0],x_shape[1],x_shape[2],x_shape[3],x_shape[4])
           out = self.complex_conv(in_fold)
        else:
           #in_rot = torch.mean(torch.mean(in_rot,2).unsqueeze(1).repeat(1,self.out_channels,1) * weightNormalize1(self.new_wr), 2).view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
           in_rot = torch.mean(in_rot, 2).view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
           in_abs = torch.mean(in_abs, 2).view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
           #in_abs = torch.mean(torch.mean(in_abs,2).unsqueeze(1).repeat(1,self.out_channels,1) *  weightNormalize1(self.new_wa), 2).view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
           out = torch.cat((in_rot,in_abs),1)
        return out 


class ComplexMaxpool(nn.Module):
    def __init__(self, kern_size, stride, dilation=(1, 1), padding = (0, 0)):
        super(ComplexMaxpool, self).__init__()
        self.kern_size = kern_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]+2*self.padding[0]-self.dilation[0]*(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]+2*self.padding[1]-self.dilation[1]*(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,x_shape[2],x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride, dilation = self.dilation, padding = self.padding)(x).view(x_shape[0], x_shape[1],  x_shape[2], self.kern_size[0]*self.kern_size[1], -1)
        temporal_buckets = torch.max(temporal_buckets, 3)[0]
        return temporal_buckets.view(-1, x_shape[1], x_shape[2], out_spatial_x, out_spatial_y)

 



class ComplexConv2Deffangle4Dxyrt(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle4Dxyrt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1x = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot1y = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2x = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot2y = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot2r = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot1r = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2t = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot1t = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1x),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1y),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1r),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2r),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
 
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x = (torch.sum(out_x*(self.weight_matrix_rot2x),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y = (torch.sum(out_y*(self.weight_matrix_rot2y),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize1(self.weight_matrix_rot1t),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize1(self.weight_matrix_rot2t),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs,out_x,out_y),1)

 





class ComplexConv2Deffangle4Dxyr(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle4Dxyr, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1x = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot1y = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2x = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot2y = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot2r = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot1r = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1x),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1y),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1r),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2r),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
 
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x = (torch.sum(out_x*(self.weight_matrix_rot2x),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y = (torch.sum(out_y*(self.weight_matrix_rot2y),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize1(self.weight_matrix_rot1r),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize1(self.weight_matrix_rot2r),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs,out_x,out_y),1)

 

class ComplexConv2Deffangle4Dxymp(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle4Dxymp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rotx = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_roty = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_rot2x = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.weight_matrix_rot2y = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1]*tbr_shape[2]).unsqueeze(1).repeat(1, self.out_channels, 1)*(self.weight_matrix_rotx),2))).view(tbr_shape[0],1,out_spatial_x, out_spatial_y,self.out_channels).permute(0,1,4,2,3).contiguous()
         
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1]*tbr_shape[2]).unsqueeze(1).repeat(1, self.out_channels, 1)*(self.weight_matrix_roty),2))).view(tbr_shape[0],1,out_spatial_x, out_spatial_y,self.out_channels).permute(0,1,4,2,3).contiguous()
         
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1]*tbr_shape[2]).unsqueeze(1).repeat(1, self.out_channels, 1)*weightNormalizexy(self.weight_matrix_rotx, self.weight_matrix_roty),2))).view(tbr_shape[0],1,out_spatial_x, out_spatial_y,self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        out_abs = torch.exp((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1]*tbr_shape[2]).unsqueeze(1).repeat(1, self.out_channels, 1)*weightNormalizexy(self.weight_matrix_rotx, self.weight_matrix_roty),2))).view(tbr_shape[0],1,out_spatial_x, out_spatial_y,self.out_channels).permute(0,1,4,2,3).contiguous()
        return torch.cat((out_rot,out_abs,out_x,out_y),1)

 


class ComplexConv2Deffangle4Dxy(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, dilation = (1, 1), padding = (0, 0), drop_prob=0.0):
        super(ComplexConv2Deffangle4Dxy, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.drop_prob = drop_prob
        self.weight_matrix_rot1x = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot1y = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2x = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot2y = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]+2*self.padding[0]-self.dilation[0]*(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]+2*self.padding[1]-self.dilation[1]*(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride, dilation = self.dilation, padding = self.padding)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1x),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1y),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        #Shape: [Batch, in_channels, L], 
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalizexy(self.weight_matrix_rot1x, self.weight_matrix_rot1y),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        #[Batch, L, in_channels] -> [Batch*L, 1, in_channels] -> [Batch*L, out_channels, in_channels]
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        # [Batch*L, out_channels]
        out_rot = (torch.sum(out_rot*weightNormalizexy(self.weight_matrix_rot2x, self.weight_matrix_rot2y),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
#         out_rot = out_rot.permute(0,2,1).contiguous().unsqueeze(-1).repeat(1,1,1,self.out_channels)
#         out_rot = (torch.sum(out_rot*weightNormalizexy(self.weight_matrix_rot2x, self.weight_matrix_rot2y).transpose(1, 0),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()

        ###
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x = (torch.sum(out_x*(self.weight_matrix_rot2x),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y = (torch.sum(out_y*(self.weight_matrix_rot2y),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalizexy(self.weight_matrix_rot1x, self.weight_matrix_rot1y),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalizexy(self.weight_matrix_rot2x,self.weight_matrix_rot2y),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs,out_x,out_y),1)


class ComplexConv2Deffangle4Dhordernlog(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, dilation = (1,1), padding = (0, 0), drop_prob=0.0):
        super(ComplexConv2Deffangle4Dhordernlog, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.drop_prob = drop_prob
        self.weight_matrix_rot_11 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot_12 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot_21 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot_22 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot_31 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot_32 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]+2*self.padding[0]-self.dilation[0]*(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]+2*self.padding[1]-self.dilation[1]*(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, dilation=self.dilation, padding=self.padding, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_11),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_11),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot_11),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot1 = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
 
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x1 = (torch.sum(out_x*(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y1 = (torch.sum(out_y*(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        
        
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_21),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_21),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot_21),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot2 = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x2 = (torch.sum(out_x*(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y2 = (torch.sum(out_y*(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_31),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_31),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot_31),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot3 = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x3 = (torch.sum(out_x*(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y3 = (torch.sum(out_y*(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        

        
        
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        #temporal_buckets_abs = temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot_11,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs1 = (torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_12,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot_21,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs2 = (torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_22,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot_31,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs3 = (torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_32,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
       
        out_x = self.relu(out_x2 - out_x3) + out_x1
        out_y = self.relu(out_y2 - out_y3) + out_y1
        out_rot = self.relu(out_rot2 - out_rot3) + out_rot1
        out_abs = self.relu(out_abs2 - out_abs3) + out_abs1
        
        #out_x = torch.max(out_x2,out_x3) + out_x1
        #out_y = torch.max(out_y2,out_y3) + out_y1
        #out_rot = torch.max(out_rot2,out_rot3) + out_rot1
        #out_abs = torch.max(out_abs2,out_abs3) + out_abs1
        
        
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs,out_x,out_y),1)

 

class ComplexConv2Deffangle4Dhorder(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, dilation = (1,1), padding = (0, 0), drop_prob=0.0):
        super(ComplexConv2Deffangle4Dhorder, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.drop_prob = drop_prob
        self.weight_matrix_rot_11 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot_12 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.biasx = torch.nn.Parameter(torch.rand(3,out_channels), requires_grad=True)
        self.weight_matrix_rot_21 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot_22 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.biasy = torch.nn.Parameter(torch.rand(3,out_channels), requires_grad=True)
        self.weight_matrix_rot_31 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot_32 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.biasr = torch.nn.Parameter(torch.rand(3,out_channels), requires_grad=True)
        #self.biasa = torch.nn.Parameter(torch.rand(3,out_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]+2*self.padding[0]-self.dilation[0]*(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]+2*self.padding[1]-self.dilation[1]*(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, dilation=self.dilation, padding=self.padding, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_11),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_11),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot_11),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_rot1 = (torch.tanh(-self.biasr[0].view(1,1,1,1,-1)) * (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_rot1 = ( (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_x1 = (self.biasx[0].view(1,1,1,1,-1) + (torch.sum(out_x*(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_x1 = ((torch.sum(out_x*(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_y1 = (self.biasy[0].view(1,1,1,1,-1) + (torch.sum(out_y*(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_y1 = ((torch.sum(out_y*(self.weight_matrix_rot_12),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        
        
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_21),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_21),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot_21),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_rot2 = (torch.tanh(-self.biasr[1].view(1,1,1,1,-1))*(torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_rot2 = ((torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_x2 = (self.biasx[1].view(1,1,1,1,-1) + (torch.sum(out_x*(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_x2 = ((torch.sum(out_x*(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_y2 = (self.biasy[1].view(1,1,1,1,-1) + (torch.sum(out_y*(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_y2 = ((torch.sum(out_y*(self.weight_matrix_rot_22),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_31),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot_31),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot_31),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_rot3 = (torch.tanh(-self.biasr[2].view(1,1,1,1,-1))*(torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_rot3 = ((torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_x3 = (self.biasx[2].view(1,1,1,1,-1) + (torch.sum(out_x*(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_x3 = ((torch.sum(out_x*(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_y3 = (self.biasy[2].view(1,1,1,1,-1) + (torch.sum(out_y*(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_y3 = ((torch.sum(out_y*(self.weight_matrix_rot_32),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()

        
        
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot_11,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_abs1 = (torch.exp(-self.biasa[0]**2).view(1,1,1,1,-1)+ torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_12,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_abs1 = (torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_12,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot_21,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_abs2 = (torch.exp(-self.biasa[1]**2).view(1,1,1,1,-1) + torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_22,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_abs2 = (torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_22,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot_31,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        #out_abs3 = (torch.exp(-self.biasa[2]**2).view(1,1,1,1,-1) + torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_32,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        out_abs3 = (torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot_32,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
       
        out_x = self.relu(out_x2 - out_x3) + self.relu(out_x1)
        out_y = self.relu(out_y2 - out_y3) + self.relu(out_y1)
        out_rot = self.relu(out_rot2 - out_rot3) + self.relu(out_rot1)
        out_abs = self.relu(out_abs2 - out_abs3) + self.relu(out_abs1)
        
        #out_x = torch.max(out_x2,out_x3) + out_x1
        #out_y = torch.max(out_y2,out_y3) + out_y1
        #out_rot = torch.max(out_rot2,out_rot3) + out_rot1
        #out_abs = torch.max(out_abs2,out_abs3) + out_abs1
        
        
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs,out_x,out_y),1)

 



class ComplexConv2Deffangle4D(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, dilation = (1,1), padding = (0, 0), drop_prob=0.0):
        super(ComplexConv2Deffangle4D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.drop_prob = drop_prob
        self.weight_matrix_rot1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.bias = torch.nn.Parameter(torch.rand(out_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]+2*self.padding[0]-self.dilation[0]*(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]+2*self.padding[1]-self.dilation[1]*(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, dilation=self.dilation, padding=self.padding, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = ((torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
 
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x = ((torch.sum(out_x*(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y = ((torch.sum(out_y*(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = (torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels)).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs,out_x,out_y),1)

 



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
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
       
        tbr_shape = temporal_buckets_rot.shape 
        
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
 
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs),1)

 


class ComplexConv2Deff(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deff, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        #self.weight_matrix_abs1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, 2, 2, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0:4,...].view(x_shape[0],2,2,self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,4,...]
        
        tbr_shape = temporal_buckets_rot.shape
        tba_shape = temporal_buckets_abs.shape
        
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,4,5,1,2).contiguous().view(-1,2,2)
        eye = temporal_buckets_rot.new_ones(temporal_buckets_rot.size(-1)).diag().expand_as(temporal_buckets_rot)
        
        temporal_buckets_rot_sub = eye - temporal_buckets_rot
        temporal_buckets_rot_add = (eye + temporal_buckets_rot)
        temporal_buckets_rot_inv = b_inv(temporal_buckets_rot_add).contiguous()
        #print(b_inv(temporal_buckets_rot_add[0]), temporal_buckets_rot_inv[0])
        #print(temporal_buckets_rot_inv.shape)
        
        #Shape: [batches, 2, 2, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot_log1 = torch.bmm(temporal_buckets_rot_sub, temporal_buckets_rot_inv).view(tbr_shape[0],tbr_shape[3], tbr_shape[4],tbr_shape[5],2,2).permute(0,4,5,1,2,3)
        temporal_buckets_rot_log = temporal_buckets_rot_log1      
        
        tbrl_shape = temporal_buckets_rot_log.shape
         
        #self.weight_matrix_rot = self.weight_matrix_rot.view(1,1,1,self.out_channels, tbr_shape[3],1).repeat(tbr_shape[0],2,2,1,1,tbr_shape[4])
        
        #Shape: [batches, 2, 2, in_channels, L]
        out = (torch.sum(temporal_buckets_rot_log.permute(0,5,1,2,3,4).contiguous().view(-1,2,2,tbrl_shape[3],tbrl_shape[4])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),4)).view(tbrl_shape[0],tbrl_shape[5],2,2,tbrl_shape[3]).permute(0,2,3,4,1).contiguous()
        out_shape = out.shape
        
        #Shape: [batches, 2, 2, out_channels, L]
        out = out.permute(0,4,1,2,3).contiguous().view(-1,2,2,1,self.in_channels).repeat(1,1,1,self.out_channels,1)
        out = torch.sum(out*weightNormalize(self.weight_matrix_rot2,self.drop_prob), 4).view(out_shape[0],out_shape[4],2,2,self.out_channels).permute(0,2,3,4,1).contiguous()
        
        out_shape = out.shape
        out = out.permute(0,3,4,1,2).contiguous().view(-1,2,2)
        eye = out.new_ones(out.size(-1)).diag().expand_as(out)
        out_sub = eye-out
        out_inv = b_inv(eye+out)
        
        #Shape: [batches, 4, out_channels, out_spatial_x, out_spatial_y]
        out_rot = torch.bmm(out_sub, out_inv).view(out_shape[0], out_shape[3], out_spatial_x, out_spatial_y,4).permute(0,4,1,2,3).contiguous()
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        out_abs = out_abs
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs),1)

 


class ComplexConv2D_bias(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.):
        super(ComplexConv2D_bias, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)
        self.b = torch.nn.Parameter(torch.rand(out_channels), requires_grad=True)
        #self.weight_matrix_abs = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, 2, 2, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0:4,...].view(x_shape[0],2,2,self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,4,...]
        
        tbr_shape = temporal_buckets_rot.shape
        tba_shape = temporal_buckets_abs.shape
        
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,4,1,2).contiguous().view(-1,2,2)
        eye = temporal_buckets_rot.new_ones(temporal_buckets_rot.size(-1)).diag().expand_as(temporal_buckets_rot)
        
        temporal_buckets_rot_sub = eye - temporal_buckets_rot
        temporal_buckets_rot_add = (eye + temporal_buckets_rot)
        temporal_buckets_rot_inv = b_inv(temporal_buckets_rot_add).contiguous()
        #print(b_inv(temporal_buckets_rot_add[0]), temporal_buckets_rot_inv[0])
        #print(temporal_buckets_rot_inv.shape)
        
        #Shape: [batches, 2, 2, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot_log1 = torch.bmm(temporal_buckets_rot_sub, temporal_buckets_rot_inv).view(tbr_shape[0],tbr_shape[3], tbr_shape[4],2,2,1).permute(0,3,4,5,1,2)
        temporal_buckets_rot_log = temporal_buckets_rot_log1.repeat(1,1,1,self.out_channels,1,1)        
        
        tbrl_shape = temporal_buckets_rot_log.shape
         
        #self.weight_matrix_rot = self.weight_matrix_rot.view(1,1,1,self.out_channels, tbr_shape[3],1).repeat(tbr_shape[0],2,2,1,1,tbr_shape[4])
        
        #Shape: [batches, 2, 2, out_channels, L]
        out = (torch.sum(temporal_buckets_rot_log.permute(0,5,1,2,3,4).contiguous().view(-1,2,2,tbrl_shape[3],tbrl_shape[4])*weightNormalize(self.weight_matrix_rot,self.drop_prob),4)).view(tbrl_shape[0],tbrl_shape[5],2,2,tbrl_shape[3]).permute(0,2,3,4,1).contiguous()
        out_shape = out.shape
        
        out = out.permute(0,3,4,1,2).contiguous().view(-1,2,2)
        eye = out.new_ones(out.size(-1)).diag().expand_as(out)
        out_sub = eye-out
        out_inv = b_inv(eye+out)
        
        #Shape: [batches, 4, out_channels, out_spatial_x, out_spatial_y]
        out_rot = torch.bmm(out_sub, out_inv).view(out_shape[0], out_shape[3], out_spatial_x, out_spatial_y,4).permute(0,4,1,2,3).contiguous()
        
        #Shape: [batches, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps).unsqueeze(1).repeat(1,self.out_channels,1,1))
        
        tba_shape = temporal_buckets_abs.shape
        temporal_buckets_abs = temporal_buckets_abs+(self.b**2).view(1,-1,1,1).repeat(tba_shape[0],1,tba_shape[2],tba_shape[3])   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        out_abs = torch.exp((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot, self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).unsqueeze(1)
        out_abs_shape = out_abs.shape
        
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        out_abs = out_abs.view(out_abs_shape[0],out_abs_shape[1], out_abs_shape[2], out_spatial_x, out_spatial_y)
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs),1)

 

    
class ComplexConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(ComplexConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.weight_matrix_rot = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, 2, 2, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0:4,...].view(x_shape[0],2,2,self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,4,...]
        
        tbr_shape = temporal_buckets_rot.shape
        tba_shape = temporal_buckets_abs.shape
        
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,4,1,2).contiguous().view(-1,2,2)
        eye = temporal_buckets_rot.new_ones(temporal_buckets_rot.size(-1)).diag().expand_as(temporal_buckets_rot)
        
        temporal_buckets_rot_sub = eye - temporal_buckets_rot
        temporal_buckets_rot_add = (eye + temporal_buckets_rot)
        temporal_buckets_rot_inv = b_inv(temporal_buckets_rot_add).contiguous()
        #print(b_inv(temporal_buckets_rot_add[0]), temporal_buckets_rot_inv[0])
        #print(temporal_buckets_rot_inv.shape)
        
        #Shape: [batches, 2, 2, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot_log1 = torch.bmm(temporal_buckets_rot_sub, temporal_buckets_rot_inv).view(tbr_shape[0],tbr_shape[3], tbr_shape[4],2,2,1).permute(0,3,4,5,1,2)
        temporal_buckets_rot_log = temporal_buckets_rot_log1.repeat(1,1,1,self.out_channels,1,1)        
        
        tbrl_shape = temporal_buckets_rot_log.shape
         
        #self.weight_matrix_rot = self.weight_matrix_rot.view(1,1,1,self.out_channels, tbr_shape[3],1).repeat(tbr_shape[0],2,2,1,1,tbr_shape[4])
        
        #Shape: [batches, 2, 2, out_channels, L]
        out = (torch.sum(temporal_buckets_rot_log.permute(0,5,1,2,3,4).contiguous().view(-1,2,2,tbrl_shape[3],tbrl_shape[4])*weightNormalize(self.weight_matrix_rot),4)).view(tbrl_shape[0],tbrl_shape[5],2,2,tbrl_shape[3]).permute(0,2,3,4,1).contiguous()
        out_shape = out.shape
        
        out = out.permute(0,3,4,1,2).contiguous().view(-1,2,2)
        eye = out.new_ones(out.size(-1)).diag().expand_as(out)
        out_sub = eye-out
        out_inv = b_inv(eye+out)
        
        #Shape: [batches, 4, out_channels, out_spatial_x, out_spatial_y]
        out_rot = torch.bmm(out_sub, out_inv).view(out_shape[0], out_shape[3], out_spatial_x, out_spatial_y,4).permute(0,4,1,2,3).contiguous()
        
        #Shape: [batches, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps).unsqueeze(1).repeat(1,self.out_channels,1,1))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        out_abs = torch.exp((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).unsqueeze(1)
        out_abs_shape = out_abs.shape
        
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        out_abs = out_abs.view(out_abs_shape[0],out_abs_shape[1], out_abs_shape[2], out_spatial_x, out_spatial_y)
        #print(self.weight_matrix_rot)
        return torch.cat((out_rot,out_abs),1)

        

class ComplexConv2D_last(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(ComplexConv2D_last, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.weight_matrix_rot = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)
        #self.weight_matrix_abs = torch.nn.Parameter(torch.rand(out_channels, in_channels*kern_size[0]*kern_size[1]), requires_grad=True)

    def forward(self, x):
        x_shape = x.shape
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, 2, 2, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0:4,...].view(x_shape[0],2,2,self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,4,...]
        
        tbr_shape = temporal_buckets_rot.shape
        tba_shape = temporal_buckets_abs.shape
        
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,4,1,2).contiguous().view(-1,2,2)
        eye = temporal_buckets_rot.new_ones(temporal_buckets_rot.size(-1)).diag().expand_as(temporal_buckets_rot)
        
        temporal_buckets_rot_sub = eye - temporal_buckets_rot
        temporal_buckets_rot_add = (eye + temporal_buckets_rot)
        temporal_buckets_rot_inv = b_inv(temporal_buckets_rot_add).contiguous()
        #print(b_inv(temporal_buckets_rot_add[0]), temporal_buckets_rot_inv[0])
        #print(temporal_buckets_rot_inv.shape)
        
        #Shape: [batches, 2, 2, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_rot_log1 = torch.bmm(temporal_buckets_rot_sub, temporal_buckets_rot_inv).view(tbr_shape[0],tbr_shape[3], tbr_shape[4],2,2,1).permute(0,3,4,5,1,2)
        temporal_buckets_rot_log = temporal_buckets_rot_log1.repeat(1,1,1,self.out_channels,1,1)        
        
        tbrl_shape = temporal_buckets_rot_log.shape
         
        #self.weight_matrix_rot = self.weight_matrix_rot.view(1,1,1,self.out_channels, tbr_shape[3],1).repeat(tbr_shape[0],2,2,1,1,tbr_shape[4])
        
        #Shape: [batches, 2, 2, out_channels, L]
        out = (torch.sum(temporal_buckets_rot_log.permute(0,5,1,2,3,4).contiguous().view(-1,2,2,tbrl_shape[3],tbrl_shape[4])*weightNormalize(self.weight_matrix_rot),4)).view(tbrl_shape[0],tbrl_shape[5],2,2,tbrl_shape[3]).permute(0,2,3,4,1).contiguous()
        out_shape = out.shape
        
        
        #Shape: [batches, 4, out_channels, out_spatial_x, out_spatial_y]
        out_rot = out[:,0,1,...].view(out_shape[0], 1, out_shape[3], out_spatial_x, out_spatial_y)
        
        #Shape: [batches, out_channels, in_channels*kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps).unsqueeze(1).repeat(1,self.out_channels,1,1))
        
        tba_shape = temporal_buckets_abs.shape   
        #self.weight_matrix_abs = self.weight_matrix_abs.view(1,self.out_channels, tba_shape[1],1).repeat(tba_shape[0],1,1,tba_shape[2])
        
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).unsqueeze(1)
        out_abs_shape = out_abs.shape
        
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        out_abs = out_abs.view(out_abs_shape[0],out_abs_shape[1], out_abs_shape[2], out_spatial_x, out_spatial_y)
        return torch.cat((out_rot,out_abs),1)

 



class ComplexBN(nn.Module):
    def __init__(self):
        super(ComplexBN, self).__init__()
        #self.in_channels = in_channels
        #self.kern_size = kern_size
        #self.stride = stride
        #self.weight_matrix = torch.Tensor([1/float(kern_size[0]*kern_size[1]) for n in range(1,kern_size[0]*kern_size[1]+1)]).to(device)

    def forward(self, x):
        x_shape = x.shape
        #y = x
        #print(torch.sum(torch.isnan(y)))
        #weight_matrix_rot = self.weight_matrix_rot
        #weight_matrix_abs = self.weight_matrix_abs
        #out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        #out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] 
        x = x.view(x_shape[0],x_shape[1],-1)
        #print(torch.sum(torch.isnan(x)))
        x_real = x[:,0,:]*x[:,4,:]
        x_img = x[:,1,:]*x[:,4,:]
        m_real = torch.mean(x_real,dim=0)
        m_img = torch.mean(x_img,dim=0)
        v_re_re = torch.var(x_real,dim=0)+eps
        v_img_img = torch.var(x_img,dim=0)+eps
        v_re_img = torch.bmm((x_real - m_real).permute(1,0).contiguous().view(-1,1,x_shape[0]), (x_img - m_img).permute(1,0).contiguous().view(-1,x_shape[0],1)).squeeze(1).squeeze(1)/x_shape[0]
        #print(v_re_img)
        #print(torch.sum(torch.isnan(v_re_img)))
        det = v_re_re*v_img_img - v_re_img*v_re_img
        trace = v_re_re+v_img_img
        sdet = torch.sqrt(det)
        ttrace = torch.sqrt(sdet*2.+trace)
        v_re_re = v_re_re/(ttrace*(det+eps))
        v_re_img = v_re_img/(ttrace*(det+eps))
        v_img_img = v_img_img/(ttrace*(det+eps))
        V = torch.cat((torch.cat(((v_img_img+sdet).view(-1,1,1),-1*v_re_img.view(-1,1,1)),dim=2),torch.cat((-1*v_re_img.view(-1,1,1),(v_re_re+sdet).view(-1,1,1)),dim=2)),dim=1) 
        
        #print(V)
        #print(det)
        x_cat = torch.cat(((x_real-m_real).view(x_shape[0],-1,1), (x_img-m_img).view(x_shape[0],-1,1)),dim=2).permute(1,2,0).contiguous()
        x_normalized = torch.bmm(V,x_cat).permute(2,0,1).contiguous()
        #x_normalized = x_cat.permute(2,0,1).contiguous()
        #print(x_normalized)
        #print(x_normalized.shape)
        x_mag = torch.norm(x_normalized+eps,dim=2)
        x_cos = x_normalized[:,:,0]/(x_mag+eps)
        x_sin = x_normalized[:,:,1]/(x_mag+eps)
        y = torch.cat((x_cos.view(x_shape[0],-1,1),x_sin.view(x_shape[0],-1,1),-1.*x_sin.view(x_shape[0],-1,1),x_cos.view(x_shape[0],-1,1),x_mag.view(x_shape[0],-1,1)),dim=2).permute(0,2,1).contiguous().view(-1,5,x_shape[2],x_shape[3],x_shape[4])
        #print(torch.sum(torch.isnan(y)))
        return y


class ComplexLinearcomplex(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim):
        super(ComplexLinearcomplex, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        #self.fc1 = nn.Linear(input_dim, output_dim)

        #self.weight = torch.nn.Parameter(torch.rand(2),requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        #self.weight_rot = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        #self.weight_abs = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True) 

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2,2]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]

        x_shape = x_rot.shape
        #weights_rot = weights.view(1,x_shape[1],1,1).repeat(x_shape[0],1,2,2)
        #weights_abs = weights.view(1,x_shape[1]).repeat(x_shape[0],1)
        
        x_rot_squeezed = x_rot.view(-1,2,2)   
        eye = x_rot_squeezed.new_ones(x_rot_squeezed.size(-1)).diag().expand_as(x_rot_squeezed)
        x_rot_squeezed_sub = eye - x_rot_squeezed
        x_rot_squeezed_inv = b_inv(eye+x_rot_squeezed)
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2,2]
        x_rot_log = torch.bmm(x_rot_squeezed_sub,x_rot_squeezed_inv).view(x_shape[0],x_shape[1],2,2).permute(0,2,3,1).contiguous()
        
        #shape: [batches, 2,2]
        out = torch.sum(x_rot_log*weightNormalize1(self.weights),3)
        eye = out.new_ones(out.size(-1)).diag().expand_as(out)
        out_sub = eye-out
        out_inv = b_inv(eye+out)
        out_rot = torch.bmm(out_sub, out_inv)
        
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))    
    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def SOmetric(self, X, Y):
        term_1 = torch.bmm(b_inv(X), Y)
        eye = term_1.new_ones(term_1.size(-1)).diag().expand_as(term_1)
        term_1_sub = eye - term_1
        term_1_inv = b_inv(eye+term_1)
        term_1_log = torch.bmm(term_1_sub, term_1_inv).view(-1,4)
        return torch.norm(term_1_log, dim=1)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 5]
        #all_data_temp =x 
        #all_data_temp[:,0,...] = torch.cos(all_data_temp[:,0,...])
        #all_data_temp[:,1,...] = torch.sin(all_data_temp[:,1,...])
        #all_data_temp[:,2,...] = torch.sin(all_data_temp[:,2,...])
        #all_data_temp[:,3,...] = torch.cos(all_data_temp[:,3,...])
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
        
        all_data_rot = all_data[:,:,0:4]
        all_data_abs = all_data[:,:,4]
        
        all_shape = all_data_rot.shape
        
        M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
        
        M_rot = M_rot.unsqueeze(1).repeat(1,all_shape[1],1,1).view(-1,2,2)
        M_abs = M_abs.unsqueeze(1).repeat(1,all_shape[1])
        complex_rot = (all_data_rot.view(-1,2,2)*b_inv(M_rot)).view(-1,all_shape[1],4)
        complex_abs = all_data_abs/(M_abs+eps)
        complex_out = torch.cat((complex_rot, complex_abs.unsqueeze(2)),2).view(all_data_shape[0],all_data_shape[1], all_data_shape[2], all_data_shape[3], all_data_shape[4]).permute(0,4,1,2,3).contiguous() 
        #A = complex_rot[:,:,0]*complex_abs
        #B = complex_rot[:,:,1]*complex_abs
        #dist_rot = (((torch.cos(self.weight_rot))).unsqueeze(0).repeat(all_shape[0],1) - ((complex_rot[:,:,0])))**2
        #dist_abs = ((self.weight_abs**2).unsqueeze(0).repeat(all_shape[0],1) - complex_abs)**2		
        #dist_rot = (self.weight_rot.unsqueeze(0).repeat(all_shape[0],1) - A)**2
        #dist_abs = (self.weight_abs.unsqueeze(0).repeat(all_shape[0],1)- B)**2
        #dist_l1 = self.weight[0]**2*dist_rot + self.weight[1]**2*dist_abs
        #fc_out = self.fc1(dist_l1)
        #print(self.weight**2)
        #print(fc_out.shape)
        #return fc_out
        return complex_out


class ComplexLinearangle1(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearangle1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        if out_dim1 > 0:
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ELU()
        else:
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1)
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))    
    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_shape = all_data_rot.shape
           
        M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
        
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
        else:
              fc_out = self.fc1(dist_l1)
        return fc_out


class ComplexLinearangle4Dnosum(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearangle4Dnosum, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        self.mp = nn.MaxPool1d(4)
        if out_dim1 > 0:
           #self.fc1 = nn.Linear(input_dim*4, input_dim)
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ReLU()
        else:
           #self.fc1 = nn.Linear(input_dim*4, input_dim)
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        
        out_x = torch.sum(x_x*(self.weights),1) + self.bias[2]
        out_y = torch.sum(x_y*(self.weights),1) + self.bias[3]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).repeat(1, all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        #dist_l1 = torch.cat((((self.weight[0]**2)*dist_rot + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y).unsqueeze(2), ((self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y).unsqueeze(2)), 2)
        dist_l1 = torch.cat((dist_rot.unsqueeze(2), dist_abs.unsqueeze(2), dist_x.unsqueeze(2), dist_y.unsqueeze(2)), 2)
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(self.mp(dist_l1).squeeze(2))))
        else:
              fc_out = self.fc1(self.mp(dist_l1).squeeze(2))
        return fc_out


class ComplexLinearangle4Dmwnlog(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearangle4Dmwnlog, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        if out_dim1 > 0:
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ReLU()
        else:
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weightsx = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weightsy = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        
        out_x = torch.sum(x_x*(self.weightsx),1) + self.bias[2]
        out_y = torch.sum(x_y*(self.weightsy),1) + self.bias[3]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = x_abs
        out_abs = (torch.sum(x_abs_log*weightNormalize1(self.weights),1))+self.bias[1]    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(X-Y)

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).repeat(1, all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
        else:
              fc_out = self.fc1(dist_l1)
        return fc_out

class ComplexLinearangle4Dmw_outfield(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim):
        super(ComplexLinearangle4Dmw_outfield, self).__init__()
        self.input_dim = input_dim
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weightsx = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weightsy = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        out_x = torch.sum(x_x*(self.weightsx),1) + self.bias[2]
        out_y = torch.sum(x_y*(self.weightsy),1) + self.bias[3]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).repeat(1, all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y
        dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        return dist_l1


    
class TensorRing(nn.Module):
    # For W = C1*k*k*C2, can do W(i1, i2, i3, i4) = tr(A1(i1), A2(i2), A3(i3), A4(i4))
    
    def __init__(self, c_in, k, c_out, core_size):
        super(TensorRing, self).__init__()
        self.in_channels
        self.A1 = torch.nn.Parameter(torch.rand(c_in, core_size, core_size), requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand(k, core_size, core_size), requires_grad=True)
        self.A3 = torch.nn.Parameter(torch.rand(k, core_size, core_size), requires_grad=True)
        self.A4 = torch.nn.Parameter(torch.rand(c_out, core_size, core_size), requires_grad=True)
        
    def forward(self, x):
        
        #Input is [B, in, x, y]
        i1 = torch.arange(1, c_in)
        i2 = torch.arange(1, k)
        i3 = torch.arange(1, k)
        i4 = torch.arange(1, c_out)
        
        
        
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels*self.kern_size[0]*self.kern_size[1], -1)
        
        
        
    
    
class ComplexLinearangle2Dmw_outfield(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim):
        super(ComplexLinearangle2Dmw_outfield, self).__init__()
        self.input_dim = input_dim
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
              
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
        dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        return dist_l1



class ComplexLinearangle4Dmw(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearangle4Dmw, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        if out_dim1 > 0:
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ReLU()
        else:
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weightsx = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weightsy = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        
        out_x = torch.sum(x_x*(self.weightsx),1) + self.bias[2]
        out_y = torch.sum(x_y*(self.weightsy),1) + self.bias[3]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).repeat(1, all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
        else:
              fc_out = self.fc1(dist_l1)
        return fc_out






class ComplexLinearangle4D(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearangle4D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        if out_dim1 > 0:
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ReLU()
        else:
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        
        out_x = torch.sum(x_x*(self.weights),1) + self.bias[2]
        out_y = torch.sum(x_y*(self.weights),1) + self.bias[3]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).repeat(1, all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
        else:
              fc_out = self.fc1(dist_l1)
        return fc_out






class ComplexLinearanglenew4D(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearanglenew4D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        if out_dim1 > 0:
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ELU()
        else:
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([3]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim, input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        
        out_x = torch.sum(x_x.unsqueeze(1).repeat(1,self.input_dim,1)*(self.weights),2)
        out_y = torch.sum(x_y.unsqueeze(1).repeat(1,self.input_dim,1)*(self.weights),2)
        out_rot = torch.sum(x_rot.unsqueeze(1).repeat(1,self.input_dim,1)*weightNormalize1(self.weights),2)
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log.unsqueeze(1).repeat(1,self.input_dim,1)*weightNormalize1(self.weights),2))    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[2]**2)*dist_y
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
        else:
              fc_out = self.fc1(dist_l1)
        return fc_out






class ComplexLinearanglenew(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim):
        super(ComplexLinearanglenew, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        if out_dim1 > 0:
           self.fc1 = nn.Linear(input_dim, out_dim1)
           self.fc2 = nn.Linear(out_dim1, output_dim)
           self.relu = nn.ELU()
        else:
           self.fc1 = nn.Linear(input_dim, output_dim)
        
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim, input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        out_rot = torch.sum(x_rot.unsqueeze(1).repeat(1,self.input_dim,1)*weightNormalize1(self.weights),2)
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log.unsqueeze(1).repeat(1,self.input_dim,1)*weightNormalize1(self.weights),2))    
    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
        if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
        else:
              fc_out = self.fc1(dist_l1)
        return fc_out



class ComplexLinearangle(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, num_cluster, out_dim1, output_dim, ord_fc=0):
        super(ComplexLinearangle, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        self.num_cluster = num_cluster
        self.ord_fc = ord_fc
        if ord_fc == 0:
           if out_dim1 > 0:
              self.fc1 = nn.Linear(input_dim*self.num_cluster, out_dim1)
              self.fc2 = nn.Linear(out_dim1, output_dim)
              self.relu = nn.ELU()
           else:
              self.fc1 = nn.Linear(input_dim*self.num_cluster, output_dim)
        else:
           if out_dim1 > 0:
              self.fc1 = nn.Linear(input_dim*2, out_dim1)
              self.fc2 = nn.Linear(out_dim1, output_dim)
              self.relu = nn.ELU()
           else:
              self.fc1 = nn.Linear(input_dim*2, output_dim)
        self.manifoldreluv2 = manifoldReLUv2angle(num_cluster)
        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(self.num_cluster, input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2,2]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        out_rot = torch.sum(x_rot.unsqueeze(1).repeat(1,self.num_cluster,1)*weightNormalize1(self.weights),2)
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log.unsqueeze(1).repeat(1,self.num_cluster,1)*weightNormalize1(self.weights),2))    
    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 5]
        #all_data_temp =x 
        #all_data_temp[:,0,...] = torch.cos(all_data_temp[:,0,...])
        #all_data_temp[:,1,...] = torch.sin(all_data_temp[:,1,...])
        #all_data_temp[:,2,...] = torch.sin(all_data_temp[:,2,...])
        #all_data_temp[:,3,...] = torch.cos(all_data_temp[:,3,...])
        if self.ord_fc==0:
           all_data = x.permute(0,2,3,4,1).contiguous()
           all_data_shape = all_data.shape
           all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
           all_data_rot = all_data[:,:,0]
           all_data_abs = all_data[:,:,1]
           
           all_shape = all_data_rot.shape
           
           M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
           dist_rot = self.SOmetric(all_data_rot.unsqueeze(2).repeat(1,1,self.num_cluster).view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1],1).view(-1)).view(all_shape[0],all_shape[1]*self.num_cluster)       
           M_ = self.manifoldreluv2(torch.cat((M_rot.unsqueeze(1).unsqueeze(3).unsqueeze(4),M_abs.unsqueeze(1).unsqueeze(3).unsqueeze(4)),1))
           M_rot = M_[:,0,:,0,0]
           M_abs = M_[:,1,:,0,0]
           dist_abs = self.P1metric(all_data_abs.unsqueeze(2).repeat(1,1,self.num_cluster).view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1],1).view(-1)).view(all_shape[0],all_shape[1]*self.num_cluster)
           #print(t)
           #dist_l1 = torch.exp(-self.weight[0]**2)*dist_rot+(torch.exp(-self.weight[1]**2))*dist_abs
           dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
           #dist_l1 = dist_rot+dist_abs
           if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
           else:
              fc_out = self.fc1(dist_l1)
           #print(self.weight**2)
           #print(fc_out.shape)
           return fc_out
        else:
           all_data = torch.zeros([x.shape[0],2,x.shape[2],x.shape[3],x.shape[4]], dtype=torch.float32).cuda()
           all_data[:,0,...] = x[:,0,...]
           all_data[:,1,...] = x[:,1,...]
           #all_data[:,2:5,...] = []
           all_data = all_data.view(x.shape[0],-1)
           if self.output_dim1 > 0:
              fc_out = self.fc2(self.relu(self.fc1(all_data)))
           else:
              fc_out = self.fc1(all_data)
           return fc_out







class ComplexLinear(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim, out_dim1, output_dim, ord_fc=0):
        super(ComplexLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim1 = out_dim1
        self.ord_fc = ord_fc
        if ord_fc == 0:
           if out_dim1 > 0:
              self.fc1 = nn.Linear(input_dim, out_dim1)
              self.fc2 = nn.Linear(out_dim1, output_dim)
              self.relu = nn.ELU()
           else:
              self.fc1 = nn.Linear(input_dim, output_dim)
        else:
           if out_dim1 > 0:
              self.fc1 = nn.Linear(input_dim*2, out_dim1)
              self.fc2 = nn.Linear(out_dim1, output_dim)
              self.relu = nn.ELU()
           else:
              self.fc1 = nn.Linear(input_dim*2, output_dim)

        #self.weight = torch.nn.Parameter(torch.FloatTensor([4.0, 0.0]),requires_grad=True)
        self.weight = torch.nn.Parameter(torch.rand([2]), requires_grad=True)
        #self.weight[1] = 3. 
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2,2]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]

        x_shape = x_rot.shape
        #weights_rot = weights.view(1,x_shape[1],1,1).repeat(x_shape[0],1,2,2)
        #weights_abs = weights.view(1,x_shape[1]).repeat(x_shape[0],1)
        
        x_rot_squeezed = x_rot.view(-1,2,2)   
        eye = x_rot_squeezed.new_ones(x_rot_squeezed.size(-1)).diag().expand_as(x_rot_squeezed)
        x_rot_squeezed_sub = eye - x_rot_squeezed
        x_rot_squeezed_inv = b_inv(eye+x_rot_squeezed)
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2,2]
        x_rot_log = torch.bmm(x_rot_squeezed_sub,x_rot_squeezed_inv).view(x_shape[0],x_shape[1],2,2).permute(0,2,3,1).contiguous()
        
        #shape: [batches, 2,2]
        out = torch.sum(x_rot_log*weightNormalize1(self.weights),3)
        eye = out.new_ones(out.size(-1)).diag().expand_as(out)
        out_sub = eye-out
        out_inv = b_inv(eye+out)
        out_rot = torch.bmm(out_sub, out_inv)
        
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))    
    
        return (out_rot,out_abs)

    def unweightedFMComplex(self, point_list_rot, point_list_abs):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs)
    
    def SOmetric(self, X, Y):
        term_1 = torch.bmm(b_inv(X), Y)
        eye = term_1.new_ones(term_1.size(-1)).diag().expand_as(term_1)
        term_1_sub = eye - term_1
        term_1_inv = b_inv(eye+term_1)
        term_1_log = torch.bmm(term_1_sub, term_1_inv).view(-1,4)
        return torch.norm(term_1_log, dim=1)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 5]
        #all_data_temp =x 
        #all_data_temp[:,0,...] = torch.cos(all_data_temp[:,0,...])
        #all_data_temp[:,1,...] = torch.sin(all_data_temp[:,1,...])
        #all_data_temp[:,2,...] = torch.sin(all_data_temp[:,2,...])
        #all_data_temp[:,3,...] = torch.cos(all_data_temp[:,3,...])
        if self.ord_fc==0:
           all_data = x.permute(0,2,3,4,1).contiguous()
           all_data_shape = all_data.shape
           all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
           all_data_rot = all_data[:,:,0:4]
           all_data_abs = all_data[:,:,4]
           
           all_shape = all_data_rot.shape
           
           M_rot, M_abs = self.unweightedFMComplex(all_data_rot, all_data_abs)
           dist_rot = self.SOmetric(all_data_rot.view(-1,2,2), M_rot.unsqueeze(1).repeat(1,all_shape[1],1,1).view(-1,2,2)).view(all_shape[0],all_shape[1])       
           dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
           #print(t)
           #dist_l1 = torch.exp(-self.weight[0]**2)*dist_rot+(torch.exp(-self.weight[1]**2))*dist_abs
           dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs
           #dist_l1 = dist_rot+dist_abs
           if self.output_dim1>0:
              fc_out = self.fc2(self.relu(self.fc1(dist_l1)))
           else:
              fc_out = self.fc1(dist_l1)
           #print(self.weight**2)
           #print(fc_out.shape)
           return fc_out
        else:
           all_data = torch.zeros([x.shape[0],2,x.shape[2],x.shape[3],x.shape[4]], dtype=torch.float32).cuda()
           all_data[:,0,...] = x[:,0,...]*x[:,4,...]
           all_data[:,1,...] = x[:,1,...]*x[:,4,...]
           #all_data[:,2:5,...] = []
           all_data = all_data.view(x.shape[0],-1)
           if self.output_dim1 > 0:
              fc_out = self.fc2(self.relu(self.fc1(all_data)))
           else:
              fc_out = self.fc1(all_data)
           return fc_out

