import torch
import argparse
import torch
from utils import save_model_by_name, h_A, DeterministicWarmup, c_dataset, reconstruction_loss, kl_divergence, save_DAG, label_traverse, save_imgsets
import os
from model import tuningfork_vae
import argparse
from torchvision.utils import save_image
import random
import copy
from tqdm import trange
import mictools
    

def main_worker(args):
    torch.autograd.set_detect_anomaly(True)
    model_name = f'{args.sup}_ecg_z{args.z_dim}_c{args.concept}_lr_{args.lr}_labelbeta_{args.labelbeta}_dagweights_{args.dag_w1}_{args.dag_w2}'
    args.device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
    model = tuningfork_vae(name=model_name, z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(args.device)
    
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)


    train_loader = c_dataset(os.path.join(args.data_path, 'train'), args.batch_size, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_path, 'test'), 1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
    

    for epoch in trange(args.pretrain_epoch):

        for idx, (img, gt ) in enumerate(train_loader):
            img = img.to(args.device) # bs x 4 x 96 x 96

            optimizer.zero_grad()
            label_rec_loss, label_kl_loss, label_recon_img ,label= model(img, gt, beta=args.beta, info= args.sup,stage=0)
            loss_pre = label_rec_loss + args.labelbeta* label_kl_loss
            loss_pre.backward()
            optimizer.step()

    
    for epoch in trange(args.epoch):

        # total_loss = 0
        # total_finkl = 0
        # total_finrec = 0
        # total_labelkl = 0
        # total_labelrec = 0
        # total_h_a=0
        # h_a = 0

        for idx, (img, gt ) in enumerate(train_loader):
            img = img.to(args.device) # bs x 4 x 96 x 96
            
            #stage 0
            optimizer.zero_grad()
            label_rec_loss, label_kl_loss, label_recon_img ,label= model(img, gt, beta=args.beta, info= args.sup,stage=0, pretrain=True)
            dag_param = model.dag.A # 4 x 4
            h_a0 = h_A(dag_param, dag_param.size()[0])
            
            loss0 = label_rec_loss + args.labelbeta* label_kl_loss + args.dag_w1 * h_a0 + args.dag_w2 *h_a0*h_a0 
            loss0.backward()
            optimizer.step()
            
            #stage 1
            optimizer.zero_grad()
            c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img, gt, beta=args.beta, info= args.sup,stage=1)
            dag_param = model.dag.A # 4 x 4
            h_a1 = h_A(dag_param, dag_param.size()[0])
            loss1 = c_kl_loss + c_rec_loss + mask_loss + args.dag_w1*h_a1 + args.dag_w2 *h_a1*h_a1
            loss1.backward()
            optimizer.step()
            
            # total_loss += loss.item()
            # total_finkl += fin_kl.item() 
            # total_labelkl += label_total_kld.item()
            
            # total_finrec += fin_rec_loss.item() 
            # total_labelrec += label_rec_loss.item()
            
            # total_h_a += h_a.item()

            m = len(train_loader)
        
        #per batch    
        if epoch % args.iter_show == 0:
            print('Epoch :'+str(epoch+1)+' total loss:'+str(total_loss/m)+'\n'\
                'final kl:'+str(total_finkl/m)+' final rec:'+str(total_finrec/m)+' DAGness:'+str(total_h_a/m)+'\n'\
                'label kl:'+str(total_labelkl/m)+' label rec:'+str(total_labelrec/m))  

        if epoch % args.iter_save == 0:
            
            if not os.path.exists(os.path.join(args.output_dir,f'epoch{epoch}')): 
                os.makedirs(os.path.join(args.output_dir,model_name, f'epoch{epoch}'))
                
            save_imgsets([img[0], final_recon[0], label_recon_img[0]], os.path.join(args.output_dir,model_name, f'epoch{epoch}',f'reconstructed_imgs_epoch{epoch}.png'))
            save_DAG(model.dag.A, os.path.join(args.output_dir,model_name,f'epoch{epoch}',f'A_epoch{epoch}'))
            save_model_by_name(model, epoch)
            label_traverse(args, epoch, model,model_name, test_loader)
            # label traverse 저장

    model.eval()
    save_DAG(model.dag.A, f'A_final')
    # label traverse 저장
    if not os.path.exists('./figs_test_vae_pendulum/'): 
        os.makedirs('./figs_test_vae_pendulum/')
        
    
    count = 0
    sample = False
        
    for idx, (img, gt) in enumerate(test_loader):
        img = img.to(args.device) # bs x 4 x 96 x 96
        for i in range(4):
            for j in range(-5,5):
                with torch.no_grad():
                    fin_kl, fin_rec_loss, mask_loss, label_rec_loss, label, label_total_kld, final_recon, label_recon_img = model(img,mask=i, sample=sample, adj=j*0, beta=args.beta)

            save_image(final_recon[0], 'figs_test_vae_pendulum/reconstructed_image_{}_{}.png'.format(i, count),  range = (0,1)) 
        save_image(img[0], './figs_test_vae_pendulum/true_{}.png'.format(count)) 
        count += 1
        if count == 10:
            break
    #mic = subprocess.call('mictools')
    #tic = subprocess.call('mictools')

    
    
        
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data
    parser.add_argument('--data_path', type=str, default='/mnt/hazel/data/causal_data/pendulum')
    parser.add_argument('--pretrain_epoch', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iter_save',   type=int, default=10, help="Save model every n epochs")
    parser.add_argument('--iter_show',   type=int, default=10, help="show loss every n epochs")
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--output_dir',default='/mnt/hazel/codes/scvae_integrate/results', type=str, help='path to save results')

    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--concept', default = 4, type= int)
    parser.add_argument('--z2_dim', default=4, type =int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    # need tuning
    parser.add_argument('--labelbeta', default=10, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--dag_w1', default=6, type=float)
    parser.add_argument('--dag_w2', default=1, type=float)
    
    
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--decoder_dist', default='bernoulli', choices=['bernoulli','gaussian'])
    
    # unsup : causalVAE w/o label
    # selfsup : mine
    # weaksup : causalVAE
    parser.add_argument('--sup', default='selfsup', choices=['unsup', 'selfsup', 'weaksup']) # currently unsup unavailable
    




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