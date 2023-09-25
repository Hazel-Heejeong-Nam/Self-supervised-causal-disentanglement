import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg') 



device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
bce = torch.nn.BCEWithLogitsLoss(reduction='none')
bce3 =  torch.nn.BCELoss(reduction='none')

def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def save_model_by_name(model, static, name):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, f'model_{name}.pt')
	state = {'state_dict': model.state_dict(), 'static': static} 
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
 


def gaussian_parameters(h, dim=-1):

	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def log_bernoulli_with_logits(x, logits):

	log_prob = -bce(input=logits, target=x).sum(-1)
	return log_prob

def condition_prior(scale, label, dim):
	mean = torch.ones(label.size()[0],label.size()[1], dim)
	var = torch.ones(label.size()[0],label.size()[1], dim)
	for i in range(label.size()[0]):
		for j in range(label.size()[1]):
			mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
			mean[i][j] = torch.ones(dim)*mul
			var[i][j] = torch.ones(dim)*1
	return mean, var


def conditional_sample_gaussian(m,v):
	#64*3*4
	sample = torch.randn(m.size()).to(device)
	z = m + (v**0.5)*sample
	return z


def sample_gaussian(m, v):
	sample = torch.randn(m.shape).to(device)
	

	z = m + (v**0.5)*sample
	return z