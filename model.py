#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from utils import log_bernoulli_with_logits, condition_prior, conditional_sample_gaussian, kl_normal, sample_gaussian, kl_divergence
from utils import Encoder_share, Decoder_DAG, DagLayer, Attention, MaskLayer, Encoder_label
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
from torch.autograd import Variable



class tuningfork_vae(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference = False, alpha=0.3):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 4
        self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        self.enc_share = Encoder_share(self.z_dim, self.channel)
        self.dec = Decoder_DAG(z_dim = self.z_dim, concept = self.z1_dim, z2_dim = self.z2_dim, channel = self.channel, y_dim=0)
        self.dag = DagLayer(self.z1_dim, self.z1_dim, i = inference)
        self.enc_label = Encoder_label(z_dim= self.z_dim, concept=self.z1_dim)

        #self.cause = nn.CausalLayer(self.z_dim, self.z1_dim, self.z2_dim)
        self.attn = Attention(self.z1_dim)
        self.mask_z = MaskLayer(self.z_dim)
        self.mask_u = MaskLayer(self.z1_dim,z1_dim=1)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)


    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
    
    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def forward(self, x, gt, mask = None, sample = False, adj = None, alpha=0.3, beta=1, info='selfsup', stage=0, pretrain=False, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction terms
        """

        q_m, q_v = self.enc_share(x)
        
        if info =='selfsup':
            labelmu, labelvar = self.enc_label(q_m)
            label_kld = kl_divergence(labelmu, labelvar)
            
            label = self.reparametrize(labelmu, labelvar) # bs x concept
            label_recon_img = self.dec.decode_label(label).reshape(x.size())
            labelrec_loss = F.binary_cross_entropy_with_logits(label_recon_img.reshape(x.size()), x, size_average=False).div(x.shape[0])
            #labelrec_loss = log_bernoulli_with_logits(x, label_recon_img.reshape(x.size()))
            #labelrec_loss = -torch.mean(labelrec_loss)
        else : 
            label = gt
            labelrec_loss=0
            label_recon_img=0
            label_kld=0
            
        if pretrain:
            return labelrec_loss, label_kld, label_recon_img ,label
        
        
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)
        
        # GRADIENT CTRL #
        # stage 0 : mask inference 전에 encoder output gradient 끊어버림
        if stage == 0:
            q_m_clone = q_m.detach()
            q_v_clone = q_v.detach()
            label_clone = label
        elif stage==1 :
            q_m_clone = q_m
            q_v_clone = q_v
            label_clone = label.detach()
        else :
            ValueError("Invalid stage encountered")
        ###
        
        decode_m, decode_v = self.dag.calculate_dag(q_m_clone.to(device), torch.ones(q_m_clone.size()[0], self.z1_dim,self.z2_dim).to(device))
        decode_m, decode_v = decode_m.reshape([q_m_clone.size()[0], self.z1_dim,self.z2_dim]),decode_v
        if sample == False:
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m_clone.size()[0], self.z1_dim,self.z2_dim).to(device)*adj
              decode_m[:, mask, :] = z_mask[:, mask, :]
              decode_v[:, mask, :] = z_mask[:, mask, :]
          m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape([q_m_clone.size()[0], self.z1_dim,self.z2_dim]),decode_v.reshape([q_m_clone.size()[0], self.z1_dim,self.z2_dim])
          
          # no label related operation if unsupervised
          if info != 'unsup':
            m_u = self.dag.mask_u(label_clone.to(device))
            g_u = self.mask_u.mix(m_u).to(device)
            u_MSE = torch.nn.MSELoss()
            u_loss = u_MSE(g_u, label_clone.float().to(device))
          else :
            u_loss = 0
            
          f_z = self.mask_z.mix(m_zm).reshape([q_m_clone.size()[0], self.z1_dim,self.z2_dim]).to(device)
          e_tilde = self.attn.attention(decode_m.reshape([q_m_clone.size()[0], self.z1_dim,self.z2_dim]).to(device),q_m_clone.reshape([q_m_clone.size()[0], self.z1_dim,self.z2_dim]).to(device))[0]
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m_clone.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              e_tilde[:, mask, :] = z_mask[:, mask, :]
              
          f_z1 = f_z+e_tilde
          if mask!= None and mask == 2 :
              z_mask = torch.ones(q_m_clone.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          if mask!= None and mask == 3 :
              z_mask = torch.ones(q_m_clone.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]

          z_given_dag = conditional_sample_gaussian(f_z1, m_zv*lambdav)
        
        decoded_bernoulli_logits,x1,x2,x3,x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label_clone.to(device))
        
        finrec_loss = log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        finrec_loss = -torch.mean(finrec_loss)

        p_m, p_v = torch.zeros(q_m_clone.size()), torch.ones(q_m_clone.size())
        cp_m, cp_v = condition_prior(self.scale, label_clone, self.z2_dim)
        cp_v = torch.ones([q_m_clone.size()[0],self.z1_dim,self.z2_dim]).to(device)
        cp_z = conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        kl = alpha*kl_normal(q_m_clone.view(-1,self.z_dim).to(device), q_v_clone.view(-1,self.z_dim).to(device), p_m.view(-1,self.z_dim).to(device), p_v.view(-1,self.z_dim).to(device))
        
        for i in range(self.z1_dim):
            kl = kl + beta*kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(device)
        mask_kl2 = torch.zeros(1).to(device)

        for i in range(4):
            mask_kl = mask_kl + 1*kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        
        mask_l = torch.mean(mask_kl) + u_loss
        
        final_recon = decoded_bernoulli_logits.reshape(x.size())
    
        if stage==0 :
            return labelrec_loss, label_kld, label_recon_img ,label_clone
        
        elif stage ==1 :
            return finrec_loss, kl, final_recon, mask_l 


