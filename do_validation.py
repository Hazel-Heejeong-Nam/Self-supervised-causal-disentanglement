import torch
import argparse
import torch
from utils import save_model_by_name, h_A, DeterministicWarmup, c_dataset, reconstruction_loss, kl_divergence, save_DAG, label_traverse, save_imgsets, permute_dims
import os
from model import tuningfork_vae, Discriminator
import argparse
from torchvision.utils import save_image
import random
import copy
from tqdm import trange
#import mictools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import pretrain, train
import datetime

def main_worker(args):
    args.device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    model = tuningfork_vae(name=None, z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(args.device)

    assert args.checkpoint !=None
    checkpoint = torch.load(args.checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert missing_keys == [] and unexpected_keys == []




    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    train_loader = c_dataset(os.path.join(args.data_root,args.dataset, 'train'), args.batch_size, args.num_workers, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_root,args.dataset, 'test'), 1,args.num_workers, shuffle=False)
   
    model.eval()

    sample = False
    fig, ax = plt.subplots(ncols=10, nrows=args.concept+1, figsize=(40,15))  
    plt.subplots_adjust(wspace=0, hspace=0)
    for idx, (img, gt) in enumerate(test_loader):
        img = img.to(args.device) # bs x 4 x 96 x 96
        for i in range(args.concept):
            j=2
            with torch.no_grad():
                c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img,gt, checkpoint['static'], mask=i, sample=sample, adj=j, beta=args.c_beta, info= args.sup,stage=1, mask_loc=2)

            ax[i][idx].imshow(c_recon_img[0].squeeze(0).detach().cpu().permute(1,2,0))
            ax[i][idx].get_xaxis().set_visible(False)
            ax[i][idx].get_yaxis().set_visible(False)
        ax[args.concept][idx].imshow(img[0].squeeze(0).detach().cpu().permute(1,2,0))
        ax[args.concept][idx].get_xaxis().set_visible(False)
        ax[args.concept][idx].get_yaxis().set_visible(False)
        if idx == 9:
            break

    plt.savefig('do_operation.png')
    #mic = subprocess.call('mictools')
    #tic = subprocess.call('mictools')

    
    
        
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    parser.add_argument('--checkpoint', type=str, default='checkpoints/08242023_selfsup_data_pendulum_z16_c4_lr_0.0001_labelbeta_20_epoch_200_dagweights_12_2_3_0.5/model_trained.pt')


    # data
    parser.add_argument('--data_root', type=str, default='/home/work/YAI-Summer/hazel/data/causal_data')
    parser.add_argument('--dataset', default='pendulum', type=str, help='pendulum | flow_noise')
    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iter_show',   type=int, default=20, help="SCVAE : Save & Show every n epochs")
    parser.add_argument('--pre_iter_show',   type=int, default=20, help="FactorVAE : Save & Show every n epochs")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir',default='/home/work/YAI-Summer/hazel/codes/scvae/results', type=str, help='path to save results')
    # data attribute
    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--concept', default = 4, type= int)
    parser.add_argument('--z2_dim', default=4, type =int)
    # model
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    # discriminator
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')
    # observer weights
    parser.add_argument('--l_beta', default=20, type=float, help='KL weight for label observer') #### key
    parser.add_argument('--l_gamma', default=10, type=float, help='gamma parameter label tc loss')
    parser.add_argument('--l_dag_w1', default=12, type=float)
    parser.add_argument('--l_dag_w2', default=2, type=float)
    # interpreter weights 
    parser.add_argument('--c_beta', default=10, type=float, help='KL weight for causality interpreter') #### key
    parser.add_argument('--c_dag_w1', default=3, type=float)
    parser.add_argument('--c_dag_w2', default=0.5, type=float)
    # options
    parser.add_argument('--decoder_dist', default='bernoulli', choices=['bernoulli','gaussian'])
    parser.add_argument('--sup', default='selfsup', help='unsup : causalVAE w/o label , selfsup : scvae, weaksup : causalVAE',choices=['unsup', 'selfsup', 'weaksup']) 
    # evaluate
    parser.add_argument('--metric', type=str, default = 'label',choices=['betavae','factorvae','label','do'])
    parser.add_argument('--gt_path', type=str, default='/home/work/YAI-Summer/hazel/data/causal_data/pendulum')
    parser.add_argument('--model_path', type=str, default='/home/work/YAI-Summer/hazel/codes/scvae/checkpoints')
    parser.add_argument('--model_name', type=str, default='selfsup_ecg_z16_c4_lr_0.0001_labelbeta_20_epoch_250_dagweights_12_2_3_0.5')
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_eval', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=5)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #seed 
    random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)

    
    #arg parsing
    args = parse_args()
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    
    main_worker(args)