import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.autograd import Variable
import subprocess
import torch.nn.functional as F
import os
from torchvision.utils import save_image
import random
matplotlib.use('Agg') 

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

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
    A = np.array(A.detach().cpu())
    Around = np.round(A)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].matshow(A)
    ax[1].matshow(Around)
    #fig.colorbar(A)
    for (i,j) , z in np.ndenumerate(A):
        ax[0].text(j,i, '{:0.3f}'.format(z), ha = 'center', va='center')
    plt.savefig(f'{name}.png')
    
    
def label_traverse(args, epoch, static, model, model_name, loader, pretrain, loc=-1):
    model.eval()

    # for label traversing
    lowlimit = static[:,0] - 2*static[:,1] # 4,
    uplimit = static[:,0] + 2*static[:,1] # 4, 
    inter = 2*static[:,1] / 9 # 4, 

    interpolation = [torch.arange(lowlimit[i], uplimit[i]+0.1, inter[i]) for i in range(lowlimit.size(0))]
    ###
    n_dsets = len(loader)
    rand_idx = random.randint(1, n_dsets-1)
    fixed_idx = 0
    
    random_img,_ = loader.dataset.__getitem__(rand_idx)
    fixed_img,_ = loader.dataset.__getitem__(fixed_idx)

    Z = {'fixed_img':fixed_img, 'random_img':random_img}
            
    gifs = []
    for key in Z.keys():
        img = Z[key].to(device)
        with torch.no_grad():
            q_m, q_v = model.enc_share(img)
            labelmu, labelvar = model.enc_label(q_m)
            z_ori = model.reparametrize(labelmu, labelvar) # bs x concept
        samples = []
        for row in range(args.concept):
            z = z_ori.clone() # z_dim
            for val in interpolation[row]:
                # 여기다 여기
                z[:, row] = val
                sample = torch.sigmoid(model.dec.decode_label(z).reshape(img.size()))
                #sample = model.dec.decode_label(z).reshape(img.size())
                gifs.append(sample)

                    
        title = 'latent_traversal(iter:{})'.format(epoch)

    if pretrain :
        output_dir = os.path.join(args.output_dir,model_name,f'pre_epoch{epoch}')
    else :
        output_dir = os.path.join(args.output_dir,model_name,f'epoch{epoch}')
        
    os.makedirs(output_dir, exist_ok=True)    
    outlen = args.z_dim//args.z2_dim #4
    gifs = torch.cat(gifs)
    gifs = gifs.reshape(2, outlen, len(interpolation[0]),4, 96, 96).transpose(1, 2)
    
    
    
    for i, key in enumerate(Z.keys()):
        os.makedirs(output_dir+'/imgs', exist_ok=True)
        for j, val in enumerate(interpolation):
            save_image(tensor=gifs[i][j].cpu(), fp=os.path.join(output_dir, 'imgs', '{}_{}.png'.format(key, j)), nrow=outlen, pad_value=1)

        grid2gif(os.path.join(output_dir, 'imgs', key+'*.png'), os.path.join(output_dir, key+'.gif'), delay=10)
        subprocess.call(['rm','-rf', os.path.join(output_dir,'imgs')])
        
        
    model.train()
    
def save_imgsets(imgset, save_path):
    true = imgset[0].detach().cpu().permute(1,2,0)
    fin = imgset[1].detach().cpu().permute(1,2,0)
    if len(imgset)==3 : 
        label = imgset[2].detach().cpu().permute(1,2,0)
    
    fig, ax = plt.subplots(ncols=len(imgset), nrows=1)
    ax[0].imshow(true)
    ax[0].set_title('True')
    ax[1].imshow(fin)
    ax[1].set_title('Final')
    if len(imgset)==3 : 
        ax[2].imshow(label)
        ax[2].set_title('label')

    plt.savefig(os.path.join(save_path, 'reconsturcted.png'))