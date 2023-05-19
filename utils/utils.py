import os
import torch
from torch import nn
from torch.nn import functional as F



device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
bce = torch.nn.BCEWithLogitsLoss(reduction='none')
bce3 =  torch.nn.BCELoss(reduction='none')

def _matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)

def h_A(A, m):
    expm_A = _matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

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

def save_model_by_name(model, global_step):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
 
class Attention(nn.Module):
  def __init__(self, in_features, bias=False):
    super().__init__()
    self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
    self.sigmd = torch.nn.Sigmoid()
    #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    #self.A = torch.zeros(in_features,in_features).to(device)
    
  def attention(self, z, e):
    a = z.matmul(self.M).matmul(e.permute(0,2,1))
    a = self.sigmd(a)
    #print(self.M)
    A = torch.softmax(a, dim = 1)
    e = torch.matmul(A,e)
    return e, A



# def _sigmoid(x):
#     I = torch.eye(x.size()[0]).to(device)
#     x = torch.inverse(I + torch.exp(-x))
#     return x

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


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

def conditional_sample_gaussian(m,v):
	#64*3*4
	sample = torch.randn(m.size()).to(device)
	z = m + (v**0.5)*sample
	return z

def kl_normal(qm, qv, pm, pv):
	"""
	Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
	sum over the last dimension

	Args:
		qm: tensor: (batch, dim): q mean
		qv: tensor: (batch, dim): q variance
		pm: tensor: (batch, dim): p mean
		pv: tensor: (batch, dim): p variance

	Return:
		kl: tensor: (batch,): kl between each sample
	"""
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl

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