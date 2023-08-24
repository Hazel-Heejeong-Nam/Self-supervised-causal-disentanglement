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



class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[S?nderby 2016]., from run_pendulum.py
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n

	def __iter__(self):
		return self

	def __next__(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t

def save_model_by_name(model, static, name):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, f'model_{name}.pt')
	state = {'state_dict': model.state_dict(), 'static': static} 
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
 

# def _sigmoid(x):
#     I = torch.eye(x.size()[0]).to(device)
#     x = torch.inverse(I + torch.exp(-x))
#     return x


def gaussian_parameters(h, dim=-1):
	"""
	Converts generic real-valued representations into mean and variance
	parameters of a Gaussian distribution

	Args:
		h: tensor: (batch, ..., dim, ...): Arbitrary tensor
		dim: int: (): Dimension along which to split the tensor for mean and
			variance

	Returns:z
		m: tensor: (batch, ..., dim / 2, ...): Mean
		v: tensor: (batch, ..., dim / 2, ...): Variance
	"""
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def log_bernoulli_with_logits(x, logits):
	"""
	Computes the log probability of a Bernoulli given its logits

	Args:
		x: tensor: (batch, dim): Observation
		logits: tensor: (batch, dim): Bernoulli logits

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""
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
# def condition_prior(scale, label, dim):
#     	mean = torch.ones(label.size()[0],label.size()[1], dim)
# 	var = torch.ones(label.size()[0],label.size()[1], dim)
# 	for i in range(label.size()[0]):
# 		for j in range(label.size()[1]):
# 			mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
# 			mean[i][j] = torch.ones(dim)*mul
# 			var[i][j] = torch.ones(dim)*1
# 	return mean, var

def conditional_sample_gaussian(m,v):
	#64*3*4
	sample = torch.randn(m.size()).to(device)
	z = m + (v**0.5)*sample
	return z


def sample_gaussian(m, v):
	"""
	Element-wise application reparameterization trick to sample from Gaussian

	Args:
		m: tensor: (batch, ...): Mean
		v: tensor: (batch, ...): Variance

	Return:
		z: tensor: (batch, ...): Samples
	"""
	################################################################################
	# TODO: Modify/complete the code here
	# Sample z
	################################################################################

	################################################################################
	# End of code modification
	################################################################################
	sample = torch.randn(m.shape).to(device)
	

	z = m + (v**0.5)*sample
	return z