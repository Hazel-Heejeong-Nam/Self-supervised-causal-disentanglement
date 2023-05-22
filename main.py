import torch
import argparse
import torch
from utils import save_model_by_name, h_A, DeterministicWarmup, c_dataset, reconstruction_loss, kl_divergence
import os
from model import tuningfork_vae
import argparse
from torchvision.utils import save_image
import random
from pretrain import pretrain_labelgen
import copy
    

def main_worker(args):
    torch.autograd.set_detect_anomaly(True)
    model_name = 'put_sth_proper'
    args.device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
    model = tuningfork_vae().to(args.device)
    if not os.path.exists('./figs_vae/'): 
        os.makedirs('./figs_vae/')


    train_loader = c_dataset(os.path.join(args.data_path, 'train'), args.batch_size, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_path, 'test'), 1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lg_lr, betas=(args.beta1, args.beta2))
    beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch


    # if args.pretrain_path != None:
    #     try:
    #         pre_model_state = torch.load(args.pretrain_path)
    #         missing_keys, unexpected_keys = pre_model.load_state_dict(pre_model_state, strict=False)
    #         assert missing_keys == [] and unexpected_keys == [] 
    #     except :
    #         print('Cannot load state dict from the given path. Start new pretraining.')
    #         pre_model = pretrain_labelgen(args, model, train_loader, optimizer)
    # else :
    #     pre_model = pretrain_labelgen(args, model, train_loader, optimizer)            

    for epoch in range(args.epoch):

        total_loss = 0
        total_rec = 0
        total_kl = 0
        h_a = 0

        for idx, (img, gt ) in enumerate(train_loader):
            img = img.to(args.device) # bs x 4 x 96 x 96
            optimizer.zero_grad()

            # 1. label inference
            loss, kl, rec, finalrecon, _, label_recon, label = model(img)

            
            dag_param = model.dag.A # 4 x 4
            h_a = h_A(dag_param, dag_param.size()[0])
            
            # 5. update causal model
            loss = loss + 3*h_a + 0.5*h_a*h_a 
            loss.backward()
            optimizer.step()
            
            
            total_loss += loss.item()
            total_kl += kl.item() 
            total_rec += rec.item() 


            m = len(train_loader)
            save_image(img[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True) 
            save_image(finalrecon[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 
            
        if epoch % 1 == 0:
            print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+'m:' + str(m))

        if epoch % args.iter_save == 0:
            save_model_by_name(model, epoch)



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data
    parser.add_argument('--data_path', type=str, default='/mnt/hazel/data/causal_data/pendulum')
    parser.add_argument('--pretrain_path', type = str)
    parser.add_argument('--pretrain_epoch', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--c_epoch', type=int, default=101,    help="Number of training epochs")

    parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
    
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    
    
    
    #pretrain model
    # causal vae joint
    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--concept', default = 4, type= int)
    parser.add_argument('--z2_dim', default=4, type =int)
    parser.add_argument('--pretrain_dec_type',  default='separate', choices=['separate','integrate'])
    parser.add_argument('--pretrain_enc_sep', default=True, type=bool, help='Use additional separated encoders if True')
    parser.add_argument('--cl_lr', default=1e-3, type=float, help='learning rate')
    # original beta vae
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--lg_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decoder_dist', default='bernoulli', choices=['bernoulli','gaussian'])
    parser.add_argument('--dag_weight', type=float, default=3)



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