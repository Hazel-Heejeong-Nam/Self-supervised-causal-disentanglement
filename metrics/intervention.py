import glob
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
import os
from utils import c_dataset

def do_op(args, model):
    
    test_loader = c_dataset(os.path.join(args.data_path, 'test'), 1,args.num_workers, shuffle=False)
    sample = False
    for j in range(-5,5):
        for idx, (img,gt) in enumerate(test_loader):
            fig, ax = plt.subplots(ncols=10, nrows=args.concept+1, figsize=(40,15))  
            plt.subplots_adjust(wspace=0, hspace=0)
            for i in range(args.concept):
                with torch.no_grad():
                    c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img,gt, mask=i, sample=sample, adj=j, beta=args.beta, info= args.sup,stage=1)
            ax[i][idx].imshow(c_recon_img[0].squeeze(0).detach().cpu().permute(1,2,0))
            ax[i][idx].get_xaxis().set_visible(False)
            ax[i][idx].get_yaxis().set_visible(False)
        ax[args.concept][idx].imshow(img[0].squeeze(0).detach().cpu().permute(1,2,0))
        ax[args.concept][idx].get_xaxis().set_visible(False)
        ax[args.concept][idx].get_yaxis().set_visible(False)
        if idx == 9:
            break
        plt.savefig(os.path.join(args.output_dir, args.model_name, 'do_operation_adj_{j}.png'))