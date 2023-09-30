import torch
import argparse
import torch
from utils import save_model_by_name, h_A, c_dataset, reconstruction_loss, kl_divergence, save_DAG, save_imgsets
from metrics import check_label
import os
from scadi import tuningfork_vae, Discriminator
import argparse
from torchvision.utils import save_image
import random
import copy
from tqdm import trange
#import mictools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import train
import datetime


def main_worker(args):
    torch.autograd.set_detect_anomaly(True)
    args.model_name = f'{datetime.date.today().strftime("%m%d%Y")}_seed{args.seed}sche_{args.schedule}_{args.sup}_data_{args.dataset}_z{args.z_dim}_c{args.concept}_obs_{args.observer}_Llr_{args.l_lr}_Clr_{args.c_lr}_labelbeta_{args.l_beta}_cbeta_{args.c_beta}_epoch_{args.epoch}_dagweights_{args.l_dag_w1}_{args.l_dag_w2}_{args.c_dag_w1}_{args.c_dag_w2}'
    args.device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    
    disc = Discriminator(z_dim = args.concept).to(args.device) if args.observer =='factorvae' else None
    model = tuningfork_vae(name=args.model_name, z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(args.device)

    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    train_loader = c_dataset(os.path.join(args.data_root,args.dataset, 'train'), args.batch_size, args.num_workers, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_root,args.dataset, 'test'), 1,args.num_workers, shuffle=False)
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=args.lr_D, betas=(args.beta1_D, args.beta2_D)) if args.observer=='factorvae' else None
    optimizer_L = torch.optim.Adam(model.parameters(), lr=args.l_lr, betas=(0.9, 0.999))
    optimizer_C = torch.optim.Adam(model.parameters(), lr=args.c_lr, betas=(0.9, 0.999))

    scheduler_L = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_L, T_max=args.epoch, eta_min=args.l_lr * 0.01) if args.schedule else None
    scheduler_C = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=args.epoch, eta_min=args.c_lr * 0.01) if args.schedule else None

    model, static = train(args, train_loader, test_loader, disc, model, optimizer_D, optimizer_L, optimizer_C, scheduler_L, scheduler_C)    
      
    model.eval()
    dag_param = model.dag.A
    save_DAG(dag_param, os.path.join(args.output_dir, args.model_name, 'A_final'))
    save_model_by_name(model,static, 'trained') # save final model

    dag_score  = h_A(dag_param, dag_param.size()[0])
    representation_function = lambda x : model.enc_label(model.enc_share(x)[0])[0]
    mean_loss, target , key = check_label(representation_function)
    

    txt_name = os.path.join(args.output_dir, f'{datetime.date.today().strftime("%m%d%Y")}_summary.txt')
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'===================== {args.model_name} =======================\n')
        f.write(args.model_name,)
        f.write(f'\nDAG SCORE: {dag_score:.6f}\n')
        f.write(f'Label Disentanglement (Average): {mean_loss:.4f}\n')
        f.write(f'Targets >>   ')
        for i in range(len(target)):
            f.write(f'{key[i]} : {target[i]}\t')
        f.write(f'\n==========================================================================================\n\n\n')
            
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data
    parser.add_argument('--data_root', type=str, default='./data/causal_data')
    parser.add_argument('--dataset', default='pendulum', type=str)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--iter_show',   type=int, default=40, help="Save & Show every n epochs")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir',default='./results', type=str, help='path to save results')
    # data attribute
    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--concept', default = 4, type= int)
    parser.add_argument('--z2_dim', default=4, type =int)

    # observer 
    parser.add_argument('--l_lr', default=0.001, type=float, help='learning rate for updating observer part')
    parser.add_argument('--l_beta', default=20, type=float, help='KL weight for label observer') #### key
    parser.add_argument('--l_gamma', default=5, type=float, help='gamma parameter label tc loss')
    parser.add_argument('--l_dag_w1', default=6, type=float)
    parser.add_argument('--l_dag_w2', default=1, type=float)
    # interpreter  
    parser.add_argument('--c_lr', default=0.0003, type=float, help='learning rate for updating interpreter part')
    parser.add_argument('--c_beta', default=4, type=float, help='KL weight for causality interpreter') #### key
    parser.add_argument('--c_dag_w1', default=3, type=float)
    parser.add_argument('--c_dag_w2', default=0.5, type=float)
    # options
    parser.add_argument('--observer', default='betavae')
    # parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    # parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    # parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--decoder_dist', default='bernoulli', choices=['bernoulli','gaussian'])
    parser.add_argument('--sup', default='unsup', help='unsup : causalVAE w/o label , selfsup : scvae, weaksup : causalVAE',choices=['unsup', 'selfsup', 'weaksup']) 
    # # evaluate
    # parser.add_argument('--metric', type=str, default = 'label',choices=['betavae','factorvae','label','do'])
    parser.add_argument('--gt_path', type=str, default='./data/causal_data/pendulum')
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_eval', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=5)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--schedule', type=bool, default=True, help='whether use CosineAnnealingWarmUpRestarts or not')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #seed 

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    #arg parsing
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    
    main_worker(args)