import torch
import argparse
import torch
from utils import save_model_by_name, h_A, DeterministicWarmup, c_dataset, reconstruction_loss, kl_divergence, save_DAG
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
    model_name = 'put_sth_proper'
    args.device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
    model = tuningfork_vae().to(args.device)
    if not os.path.exists('./figs_vae/'): 
        os.makedirs('./figs_vae/')


    train_loader = c_dataset(os.path.join(args.data_path, 'train'), args.batch_size, shuffle=True)
    test_loader = c_dataset(os.path.join(args.data_path, 'test'), 1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
    

    for epoch in trange(args.epoch):

        total_loss = 0
        total_rec = 0
        total_kl = 0
        total_h_a=0
        h_a = 0

        for idx, (img, gt ) in enumerate(train_loader):
            img = img.to(args.device) # bs x 4 x 96 x 96
            optimizer.zero_grad()

            loss, kl, rec, finalrecon, _, label_recon, label = model(img)
            dag_param = model.dag.A # 4 x 4
            h_a = h_A(dag_param, dag_param.size()[0])
            loss = loss + 3*h_a + 0.5*h_a*h_a 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_kl += kl.item() 
            total_rec += rec.item() 
            total_h_a += h_a.item()

            m = len(train_loader)
            
        if epoch % args.iter_show == 0:
            print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+' DAGness:'+str(total_h_a/m))

        if epoch % args.iter_save == 0:
            save_image(img[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True) 
            save_image(finalrecon[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 
            save_DAG(model.dag.A, f'A_epoch{epoch}')
            save_model_by_name(model, epoch)

    model.eval()
    save_DAG(model.dag.A, f'A_final')
    if not os.path.exists('./figs_test_vae_pendulum/'): 
        os.makedirs('./figs_test_vae_pendulum/')
        
    
    count = 0
    sample = False
        
    for idx, (img, gt) in enumerate(test_loader):
        img = img.to(args.device) # bs x 4 x 96 x 96
        for i in range(4):
            for j in range(-5,5):
                with torch.no_grad():
                    loss, kl, rec, finalrecon, _, label_recon, label = model(img,mask=i, sample=sample, adj=j*0)
            save_image(finalrecon[0], 'figs_test_vae_pendulum/reconstructed_image_{}_{}.png'.format(i, count),  range = (0,1)) 
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
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iter_save',   type=int, default=3, help="Save model every n epochs")
    parser.add_argument('--iter_show',   type=int, default=1, help="show loss every n epochs")
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--output_dir',default='results', type=str, help='path to save results')

    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--concept', default = 4, type= int)
    parser.add_argument('--z2_dim', default=4, type =int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE') #### key
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
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