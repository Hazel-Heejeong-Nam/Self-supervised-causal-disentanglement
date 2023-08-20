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
    

def main_worker(args):
    torch.autograd.set_detect_anomaly(True)
    args.model_name = f'{args.sup}_data_{args.dataset}_z{args.z_dim}_c{args.concept}_lr_{args.lr}_labelbeta_{args.l_beta}_epoch_{args.epoch}_dagweights_{args.l_dag_w1}_{args.l_dag_w2}_{args.c_dag_w1}_{args.c_dag_w2}'
    args.device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    
    disc = Discriminator(z_dim = args.concept).to(args.device)
    model = tuningfork_vae(name=args.model_name, z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(args.device)

    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    train_loader = c_dataset(os.path.join(args.data_root,args.dataset, 'train'), args.batch_size, args.num_workers, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_root,args.dataset, 'test'), 1,args.num_workers, shuffle=False)

    optimizer_D = torch.optim.Adam(disc.parameters(), lr=args.lr_D, betas=(args.beta1_D, args.beta2_D))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    #beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
    
    if args.sup == 'selfsup':
        print('\n\n--------------START PRETRAINING')
        disc, model= pretrain(args,train_loader,test_loader, disc, model, optimizer_D, optimizer)

    print('\n\n--------------START TRAINING')
    model = train(args, train_loader, test_loader, disc, model, optimizer)          
    model.eval()
    
    save_DAG(model.dag.A, os.path.join(args.output_dir, args.model_name, 'A_final'))
    save_model_by_name(model) # save final model
    # if not os.path.exists(os.path.join(args.output_dir, model_name, 'evals')): 
        #     os.makedirs(os.path.join(args.output_dir, model_name, 'evals'))
        
    sample = False
    fig, ax = plt.subplots(ncols=10, nrows=args.concept+1, figsize=(40,15))  
    plt.subplots_adjust(wspace=0, hspace=0)
    for idx, (img, gt) in enumerate(test_loader):
        img = img.to(args.device) # bs x 4 x 96 x 96
        for i in range(args.concept):
            for j in range(-5,5):
                with torch.no_grad():
                    c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img,gt, mask=i, sample=sample, adj=j*0, beta=args.c_beta, info= args.sup,stage=1)

            ax[i][idx].imshow(c_recon_img[0].squeeze(0).detach().cpu().permute(1,2,0))
            ax[i][idx].get_xaxis().set_visible(False)
            ax[i][idx].get_yaxis().set_visible(False)
        ax[args.concept][idx].imshow(img[0].squeeze(0).detach().cpu().permute(1,2,0))
        ax[args.concept][idx].get_xaxis().set_visible(False)
        ax[args.concept][idx].get_yaxis().set_visible(False)
        if idx == 9:
            break

    plt.savefig(os.path.join(args.output_dir, args.model_name, 'do_operation.png'))
    #mic = subprocess.call('mictools')
    #tic = subprocess.call('mictools')

    
    
        
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data
    parser.add_argument('--data_root', type=str, default='/home/work/YAI-Summer/hazel/data/causal_data')
    parser.add_argument('--dataset', default='pendulum', type=str, help='pendulum | flow_noise')
    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iter_show',   type=int, default=10, help="SCVAE : Save & Show every n epochs")
    parser.add_argument('--pre_iter_show',   type=int, default=10, help="FactorVAE : Save & Show every n epochs")
    parser.add_argument('--num_workers', type=int, default=4)
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
    parser.add_argument('--l_beta', default=10, type=float, help='KL weight for label observer') #### key
    parser.add_argument('--l_gamma', default=10, type=float, help='gamma parameter label tc loss')
    parser.add_argument('--l_dag_w1', default=12, type=float)
    parser.add_argument('--l_dag_w2', default=2, type=float)
    # interpreter weights 
    parser.add_argument('--c_beta', default=4, type=float, help='KL weight for causality interpreter') #### key
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
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    
    #arg parsing
    args = parse_args()
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    
    main_worker(args)