import torch
import argparse
import torch
from utils import save_model_by_name, h_A, DeterministicWarmup, c_dataset, reconstruction_loss, kl_divergence, save_DAG, label_traverse, save_imgsets
import os
from model import tuningfork_vae, Discriminator
import argparse
from torchvision.utils import save_image
import random
import copy
from tqdm import trange
import mictools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
    

def main_worker(args):
    torch.autograd.set_detect_anomaly(True)
    args.model_name = f'{args.sup}_ecg_z{args.z_dim}_c{args.concept}_lr_{args.lr}_labelbeta_{args.labelbeta}_epoch_{args.epoch}_dagweights_{args.l_dag_w1}_{args.l_dag_w2}_{args.dag_w1}_{args.dag_w2}'
    args.device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
    model = tuningfork_vae(name=args.model_name, z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(args.device)
    
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)


    train_loader = c_dataset(os.path.join(args.data_path, 'train'), args.batch_size, args.num_workers, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_path, 'test'), 1,args.num_workers, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
    
    if args.sup == 'selfsup':
        print('\n\n--------------START PRETRAINING')
        for epoch in trange(args.pretrain_epoch):
            pre_total = 0
            pre_total_kl = 0
            pre_total_rec = 0
            
            for idx, (img, gt ) in enumerate(train_loader):
                img = img.to(args.device) # bs x 4 x 96 x 96

                optimizer.zero_grad()
                label_rec_loss, label_kl_loss, label_recon_img ,label= model(img, gt, beta=args.beta, info= args.sup,stage=0)
                loss_pre = label_rec_loss + args.labelbeta* label_kl_loss
                loss_pre.backward()
                optimizer.step()
            
                pre_total += loss_pre.item()
                pre_total_kl += label_kl_loss.item()
                pre_total_rec += label_rec_loss.item()
                
                m = len(train_loader)
                
            if epoch % args.pre_iter_show == 0:
                print(f'Pretrain epoch {epoch+1}    total : {pre_total/m}, kl : {pre_total_kl/m}, rec : {pre_total_rec/m}')
                label_traverse(args, epoch, model,args.model_name, test_loader, pretrain=True)

    print('\n\n--------------START TRAINING')
    for epoch in trange(args.epoch):

        total_loss = 0
        total_DAG = 0
        total_c_rec = 0
        total_c_kl = 0
        total_l_rec = 0
        total_l_kl = 0
        
        for idx, (img, gt ) in enumerate(train_loader):
            img = img.to(args.device) # bs x 4 x 96 x 96
            
            if args.sup =='selfsup':
                #stage 0
                optimizer.zero_grad()
    
                
                label_rec_loss, label_kl_loss, label_recon_img ,label= model(img, gt, beta=args.beta, info= args.sup,stage=0, pretrain=True)
                dag_param = model.dag.A # 4 x 4
                h_a0 = h_A(dag_param, dag_param.size()[0])

                
                
                loss0 = label_rec_loss + args.labelbeta* label_kl_loss +  args.l_dag_w1 * h_a0 + args.l_dag_w2 *h_a0*h_a0 
                loss0.backward(retain_graph=True)
                optimizer.step()
            else : 
                loss0 = 0
                h_a0 = 0
            
            #stage 1
            optimizer.zero_grad()
            c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img, gt, beta=args.beta, info= args.sup,stage=1)
            dag_param = model.dag.A # 4 x 4
            h_a1 = h_A(dag_param, dag_param.size()[0])
            loss1 = c_kl_loss + c_rec_loss + mask_loss + args.dag_w1*h_a1 + args.dag_w2 *h_a1*h_a1
            loss1.backward()
            optimizer.step()

            # loss
            total_loss = total_loss +  loss0 + loss1
            total_DAG = total_DAG + (h_a0 + h_a1)/2
            total_c_rec += c_rec_loss.item()
            total_c_kl += c_kl_loss.item()
            if args.sup == 'selfsup':
                total_l_rec += label_rec_loss.item()
                total_l_kl += label_kl_loss.item()
            


            m = len(train_loader)
        if epoch % args.iter_show == 0:
            save_path = os.path.join(args.output_dir,args.model_name,f'epoch{epoch}')
            if not os.path.exists(save_path): 
                os.makedirs(os.path.join(args.output_dir,args.model_name, f'epoch{epoch}'))
            save_DAG(model.dag.A, os.path.join(save_path,f'A_epoch{epoch}'))
            #save_model_by_name(model, epoch)
            print(f'Epoch {epoch+1}     total loss: {total_loss.item()/m}, total DAG : {total_DAG/m}')
            print(f'                    causal recon: {total_c_rec/m}, causal kl: {total_c_kl/m}')
            
            if args.sup == 'selfsup':
                print(f'                    label recon: {total_l_rec/m}, label kl: {total_l_kl/m}')
                save_imgsets([img[0], c_recon_img[0], label_recon_img[0]], save_path)
                label_traverse(args, epoch, model,args.model_name, test_loader,pretrain=False)
            else :
                save_imgsets([img[0], c_recon_img[0]], save_path)
                
                
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
                    c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img,gt, mask=i, sample=sample, adj=j*0, beta=args.beta, info= args.sup,stage=1)

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
    parser.add_argument('--data_path', type=str, default='/mnt/hazel/data/causal_data/pendulum')
    parser.add_argument('--pretrain_epoch', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--iter_show',   type=int, default=10, help="Save model every n epochs")
    parser.add_argument('--pre_iter_show',   type=int, default=10, help="Save model every n epochs")
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--output_dir',default='/mnt/hazel/codes/scvae_integrate/results', type=str, help='path to save results')

    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--concept', default = 4, type= int)
    parser.add_argument('--z2_dim', default=4, type =int)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    # need tuning
    parser.add_argument('--labelbeta', default=20, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--dag_w1', default=3, type=float)
    parser.add_argument('--dag_w2', default=0.5, type=float)
    parser.add_argument('--l_dag_w1', default=12, type=float)
    parser.add_argument('--l_dag_w2', default=2, type=float)
    
    
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--decoder_dist', default='bernoulli', choices=['bernoulli','gaussian'])
    
    # unsup : causalVAE w/o label
    # selfsup : mine
    # weaksup : causalVAE
    parser.add_argument('--sup', default='selfsup', choices=['unsup', 'selfsup', 'weaksup']) # currently unsup unavailable
    

    # evaluate
    parser.add_argument('--metric', type=str, default = 'label',choices=['betavae','factorvae','label'])
    parser.add_argument('--gt_path', type=str, default='/mnt/hazel/data/causal_data/pendulum/test')
    parser.add_argument('--model_path', type=str, default='/mnt/hazel/codes/scvae_integrate/checkpoints')
    parser.add_argument('--model_name', type=str, default='selfsup_ecg_z16_c4_lr_0.0001_labelbeta_20_epoch_170_dagweights_12_2_3_0.5')
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_eval', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=5)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #seed 
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    
    #arg parsing
    args = parse_args()
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    
    main_worker(args)