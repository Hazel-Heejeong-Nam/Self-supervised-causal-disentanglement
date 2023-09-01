import argparse 
from metrics import betavae, factorvae, check_label, do_op
import numpy as np
from main import parse_args
import torch
from model import tuningfork_vae
import os
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


if __name__ == "__main__":
    
    args = parse_args()
    checkpoint = '/home/work/YAI-Summer/hazel/codes/new_scvae/checkpoints/08312023_selfsup_data_pendulum_z16_c4_obs_betavae_lr_3e-05_labelbeta_20_epoch_300_dagweights_12_2_3_0.5/model_trained.pt'

    ground_truth_data = args.gt_path
    
    #get representation function
    tmodel = tuningfork_vae(z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(device)
    state = torch.load(checkpoint)
    static = state['static']
    missing_keys, unexpected_keys = tmodel.load_state_dict(state['state_dict'], strict=False)
    assert missing_keys == [] and unexpected_keys == [] 
    tmodel.eval()
    #representation_function = lambda x : tmodel.reparametrize(tmodel.enc_label(tmodel.enc_share(x)[0])[0],tmodel.enc_label(tmodel.enc_share(x)[0])[1])
    representation_function = lambda x : tmodel.enc_label(tmodel.enc_share(x)[0])[0]

    #set random state
    random_state = np.random.RandomState(0)
    beta_scores = betavae(ground_truth_data, representation_function, random_state, args.eval_batch_size,args.num_train, args.num_eval)
    # factor_scores = factorvae(ground_truth_data, representation_function, random_state, args.eval_batch_size, args.num_train, args.num_eval, 500) # random_state, 5, 3000,2000, 2500)
    mean_loss, target, key = check_label(representation_function)

    #do_op(args, tmodel)
