import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import subprocess
import torch.nn.functional as F
import os
from torchvision.utils import save_image
import random


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)
    
def cuda(tensor):
    return tensor.to(torch.device('cuda:1'))
    
    
def save_DAG(A, name):
    A = np.array(A.cpu())
    fig, ax = plt.subplots()
    ax.matshow(A)
    
    for (i,j) , z in np.ndenumerate(A):
        ax.text(j,i, '{:0.3f}'.format(z), ha = 'center', va='center')
    plt.savefig(f'{name}.png')
    
    
# label에 sigmoid 해서 normalize 해둠
def vis_disentangle(args, model, limit=1, inter=0.2, loc=-1):
    model.eval()

    interpolation = torch.arange(-limit, limit+0.1, inter)

    n_dsets = len(self.data_loader.dataset)
    rand_idx = random.randint(1, n_dsets-1)

    random_img = self.data_loader.dataset.__getitem__(rand_idx)
    random_img = Variable(cuda(random_img), volatile=True).unsqueeze(0)
    random_img_z = encoder(random_img)[0]

    random_z = Variable(cuda(torch.rand(1, self.z_dim)), volatile=True)

    fixed_idx = 0
    fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
    fixed_img = Variable(cuda(fixed_img), volatile=True).unsqueeze(0)
    fixed_img_z =encoder(fixed_img)[0]

    Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

    gifs = []
    for key in Z.keys():
        z_ori = Z[key]
        samples = []

        for row in range(args.z_dim):
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                # 여기다 여기
                z[:, row] = val
                sample = F.sigmoid(decoder.decode(z)).data
                samples.append(sample)
                gifs.append(sample)

                    
        samples = torch.cat(samples, dim=0).cpu()
        title = '{}_latent_traversal(iter:{})'.format(key, epoch)


    output_dir = os.path.join(args.output_dir,epoch)
    os.makedirs(output_dir, exist_ok=True)
    outlen = args.z_dim//args.z2_dim
    gifs = torch.cat(gifs)
    gifs = gifs.view(len(Z), outlen, len(interpolation),4, 96, 96).transpose(1, 2)
    for i, key in enumerate(Z.keys()):
        for j, val in enumerate(interpolation):
            save_image(tensor=gifs[i][j].cpu(),
                        fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                        nrow=outlen, pad_value=1)

        grid2gif(os.path.join(output_dir, key+'*.jpg'),
                    os.path.join(output_dir, key+'.gif'), delay=10)
        #subprocess.call(['rm',os.path.join(output_dir,'*.jpg')])

    model.train()