import torch
import argparse
import torch
from utils import save_model_by_name, _h_A, DeterministicWarmup, c_dataset
import os
from causal_vae import CausalVAE
from label_gen import BetaVAE_H
import argparse
from torchvision.utils import save_image
import random
from pretrain import pretrain
    

def main_worker(args):
    model_name = 'put_sth_proper'
    device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
    c_model = CausalVAE(name=model_name, z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(device)
    pre_model = BetaVAE_H(args.z_dim, args.concept, args.z2_dim, args.pretrain_dec_type, enc_sep=args.pretrain_enc_sep).to(args.device)
    if not os.path.exists('./figs_vae/'): 
        os.makedirs('./figs_vae/')


    train_dataset = c_dataset(args.data_path, args.batch_size)
    test_dataset = c_dataset(args.data_path, 1)
    optimizer = torch.optim.Adam(c_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch

    if args.pretrain_path != None:
        try:
            pretrained_model = torch.load(args.pretrain_path)
        except :
            print('Cannot load state dict from the given path. Start new pretraining.')
            pretrained_model = pretrain(args.pretrain_iter)
    else :
        pretrained_model = pretrain(args.pretrain_iter)            
        
    missing_keys, unexpected_keys = c_model.enc.load_state_dict(pretrained_model.enc, strict=False)
    assert missing_keys == [] and unexpected_keys == [] 
    missing_keys2, unexpected_keys2 = pre_model.load_state_dict(pretrained_model, strict=False) # inference 할 때 label만 필요하면 이거 버려도 됨
    assert missing_keys2 == [] and unexpected_keys2 == []    
    
    pre_model.eval()  # for label inference  
    
    for epoch in range(args.c_epoch):
        c_model.train()
        total_loss = 0
        total_rec = 0
        total_kl = 0
        
        #로더 좀 써라 제발;;
        for l in train_dataset:
            l= l.to(args.device)
            # label inference
            u = pre_model.enc(l)
            pre_loss_1 = pre_model.dec(u)
            
            # causal learning
            optimizer.zero_grad()
            main_loss, kl, rec, reconstructed_image,_ = c_model.negative_elbo_bound(u,l,sample = False)
            
            dag_param = c_model.dag.A
            

            h_a = _h_A(dag_param, dag_param.size()[0])
            main_loss = main_loss + 3*h_a + 0.5*h_a*h_a 
    
    
            main_loss.backward()
            optimizer.step()
            pre_loss = pre_loss_1 + h_a
            # 여기서 따로 backprop 한다고 해도 update 의 방향이 두 가지임

            total_loss += main_loss.item()
            total_kl += kl.item() 
            total_rec += rec.item() 

            m = len(train_dataset)
            save_image(u[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True) 
            save_image(reconstructed_image[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 
            
        if epoch % 1 == 0:
            print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+'m:' + str(m))

        if epoch % args.iter_save == 0:
            save_model_by_name(c_model, epoch)



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data
    parser.add_argument('--data_path', type=str, default='./causal_data/pendulum')
    parser.add_argument('--pretrain_path', type = str)
    parser.add_argument('--pretrain_iter', type=int, default=100)
    parser.add_argument('--c_batch_size', type=int, default=64)
    parser.add_argument('--c_epoch', type=int, default=101,    help="Number of training epochs")

    parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
    
    
    #pretrain model
    # causal vae joint
    parser.add_argument('z_dim')
    parser.add_argument('concept')
    parser.add_argument('z2_dim')
    parser.add_argument('pretrain_dec_type')
    parser.add_argument('pretrain_enc_sep')
    # original beta vae
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--pre_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--decoder_dist', default='bernoulli', choices=['bernoulli','gaussian'])



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