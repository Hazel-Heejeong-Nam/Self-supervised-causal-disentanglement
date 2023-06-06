import argparse 
from metrics import betavae, factorvae, check_label
import numpy as np
from main import parse_args
import torch
from model import tuningfork_vae
import os
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")


if __name__ == "__main__":
    
    args = parse_args()
    # get ground truth data
    if args.model_name == None :
        model_name = f'{args.sup}_ecg_z{args.z_dim}_c{args.concept}_lr_{args.lr}_labelbeta_{args.labelbeta}_epoch_{args.epoch}_dagweights_{args.l_dag_w1}_{args.l_dag_w2}_{args.dag_w1}_{args.dag_w2}'
    else :
        model_name = args.model_name
        
    model_path = os.path.join(args.model_path, model_name, 'model_trained.pt')
    ground_truth_data = args.gt_path
    
    print(f'evaluate {model_name}, method : {args.metric}')
    
    #get representation function
    tmodel = tuningfork_vae(z_dim=args.z_dim, z1_dim=args.concept, z2_dim=args.z2_dim).to(device)
    state = torch.load(model_path)
    missing_keys, unexpected_keys = tmodel.load_state_dict(state, strict=False)
    assert missing_keys == [] and unexpected_keys == [] 
    tmodel.eval()
    representation_function = lambda x : tmodel.reparametrize(tmodel.enc_label(tmodel.enc_share(x)[0])[0],tmodel.enc_label(tmodel.enc_share(x)[0])[1])
    
    #set random state
    random_state = np.random.RandomState(0)
    if args.metric == 'betavae':
        scores = betavae(ground_truth_data, representation_function, random_state, args.eval_batch_size,args.num_train, args.num_eval)
    elif args.metric == 'factorvae':
        scores = factorvae(ground_truth_data, representation_function, random_state, args.eval_batch_size, args.num_train, args.num_eval, 500)
        #scores = factor_vae.compute_factor_vae(ground_truth_data, representation_function, random_state, 5, 3000,2000, 2500)
    elif args.metric == 'dds':
        pass
    elif args.metric == 'label':
        check_label(representation_function)
    else :
        ValueError('Undefined metric encountered')