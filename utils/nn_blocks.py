#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")




class MaskLayer(nn.Module):
	def __init__(self, z_dim, concept=4,z1_dim=4):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim),
		)
		self.net2 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim),
		)
		self.net3 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
		  nn.Linear(32, z1_dim),
		)
		self.net4 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim)
		)
		self.net = nn.Sequential(
			nn.Linear(z_dim , 32),
			nn.ELU(),
			nn.Linear(32, z_dim),
		)
	def masked(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z
   
	def masked_sep(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z
   
	def mix(self, z):
		zy = z.view(-1, self.concept*self.z1_dim)
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			if self.concept ==4:
				zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
			elif self.concept ==3:
				zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
		else:
			if self.concept ==4:
				zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
			elif self.concept ==3:
				zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		if self.concept ==4:
			rx4 = self.net4(zy4)
			h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
		elif self.concept ==3:
			h = torch.cat((rx1,rx2,rx3), dim=1)
		#print(h.size())
		return h
   
   
    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features,i = False, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features,out_features)
        self.a = self.a
        #self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
        #self.a[1][2], self.a[1][3] = 1,1
        self.A = nn.Parameter(self.a)
        
        self.b = torch.eye(out_features)
        self.b = self.b
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x
        
    def inv_cal(self, x,v):
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, self.I - self.A, self.bias)
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v

    def calculate_dag(self, x, v):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias) 
        #print(x.size())
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
        
   
    #def encode_
    def forward(self, x):
        x = x * torch.inverse((self.A)+self.I)
        return x
      


class Encoder(nn.Module):
	def __init__(self, z_dim, channel=4, y_dim=4):
		super().__init__()
		self.z_dim = z_dim
		self.y_dim = y_dim
		self.channel = channel
		self.fc1 = nn.Linear(self.channel*96*96, 300)
		self.fc2 = nn.Linear(300+y_dim, 300)
		self.fc3 = nn.Linear(300, 300)
		self.fc4 = nn.Linear(300, 2 * z_dim)
		self.LReLU = nn.LeakyReLU(0.2, inplace=True)
		self.net = nn.Sequential(
			nn.Linear(self.channel*96*96, 900),
			nn.ELU(),
			nn.Linear(900, 300),
			nn.ELU(),
			nn.Linear(300, 2 * z_dim),
		)


	def encode(self, x, y=None):
		xy = x if y is None else torch.cat((x, y), dim=1)
		xy = xy.view(-1, self.channel*96*96)
		h = self.net(xy)
		m, v = ut.gaussian_parameters(h, dim=1)
		#print(self.z_dim,m.size(),v.size())
		return m, v
   
   
class Decoder_DAG(nn.Module):
	def __init__(self, z_dim, concept, z1_dim, channel = 4, y_dim=0):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		self.y_dim = y_dim
		self.channel = channel
		#print(self.channel)
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net2 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net3 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net4 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net5 = nn.Sequential(
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
   
		self.net6 = nn.Sequential(
			nn.Linear(z_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)

   
	def decode_union(self, z, u, y=None):
		
		z = z.view(-1, self.concept*self.z1_dim)
		zy = z if y is None else torch.cat((z, y), dim=1)
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			zy1, zy2, zy3, zy4 = zy[:,0],zy[:,1],zy[:,2],zy[:,3]
		else:
			zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		rx4 = self.net4(zy4)
		h = self.net5((rx1+rx2+rx3+rx4)/4)
		return h,h,h,h,h
   
	def decode(self, z, u , y = None):
		z = z.view(-1, self.concept*self.z1_dim)
		h = self.net6(z)
		return h, h,h,h,h
    
	def decode_sep(self, z, u, y=None):
		z = z.view(-1, self.concept*self.z1_dim)
		zy = z if y is None else torch.cat((z, y), dim=1)
			
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			if self.concept ==4:
				zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
			elif self.concept ==3:
				zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
		else:
			if self.concept ==4:
				zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
			elif self.concept ==3:
				zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		if self.concept ==4:
			rx4 = self.net4(zy4)
			h = (rx1+rx2+rx3+rx4)/self.concept
		elif self.concept ==3:
			h = (rx1+rx2+rx3)/self.concept
		
		return h,h,h,h,h
   
	def decode_cat(self, z, u, y=None):
		z = z.view(-1, 4*4)
		zy = z if y is None else torch.cat((z, y), dim=1)
		zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		rx4 = self.net4(zy4)
		h = self.net5( torch.cat((rx1,rx2, rx3, rx4), dim=1))
		return h
   

