from utils import permute_dims
import torch
from torch.nn import functional as F
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
from utils import save_model_by_name, h_A, DeterministicWarmup, c_dataset, reconstruction_loss, kl_divergence, save_DAG, label_traverse, save_imgsets, permute_dims
import os
from model import tuningfork_vae, Discriminator
import argparse
from torchvision.utils import save_image
import random
import copy
import numpy as np
from tqdm import trange

def train(args,train_loader, test_loader,Discriminator, model, optimizer_D, optimizer_L, optimizer_C):
    if args.sup == 'selfsup':
        static = torch.Tensor([[0,1],[0,1],[0,1],[0,1]])# initial static
    else :
        static = None # 나중에는 predefined 로 바꿔줘야함


    for epoch in trange(args.epoch):
    
        total_loss, total_DAG = 0,0
        total_c_rec, total_c_kl = 0,0
        total_l_rec, total_l_kl = 0,0
        total_l_tc = 0
        label_list = []

        for idx, (img, gt ) in enumerate(train_loader):
            img = img.to(args.device) # bs x 4 x 96 x 96
            
            if args.sup =='selfsup' and args.observer =='betavae':
                #stage 0
                optimizer_L.zero_grad()
                label_rec_loss, label_kl_loss, label_recon_img ,label= model(img, gt, static, beta=args.c_beta, info= args.sup,stage=0)
                dag_param = model.dag.A # 4 x 4
                h_a0 = h_A(dag_param, dag_param.size()[0])
                loss0 = label_rec_loss + args.l_beta* label_kl_loss+ args.l_dag_w1 * h_a0 + args.l_dag_w2 *h_a0*h_a0 
                loss0.backward(retain_graph=True)
                optimizer_L.step()
                label_list = label_list + list(label.detach().cpu())
                total_l_rec += label_rec_loss.item()
                total_l_kl += label_kl_loss.item()

            elif args.sup =='selfsup' and args.observer =='factorvae':
                halflen=img.size(0)//2
                img_1, img_2 = img[:halflen], img[halflen:]
                gt_1, gt_2 = gt[:halflen], gt[halflen:]

                label_rec_loss, label_kl_loss, label_recon_img ,label= model(img_1, gt_1, static, beta=args.c_beta, info= args.sup,stage=0)
                D_z = Discriminator(label)
                label_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                loss_pre = label_rec_loss + args.l_beta* label_kl_loss + args.l_gamma * label_tc_loss + args.l_dag_w1 * h_a0 + args.l_dag_w2 *h_a0*h_a0 
                optimizer.zero_grad()
                loss_pre.backward(retain_graph=True)
                
                D_label_rec_loss, D_label_kl_loss, D_label_recon_img ,D_label= model(img_2, gt_2, None, beta=args.c_beta, info= args.sup,stage=0, pretrain=True, dec=False)
                label_perm = permute_dims(D_label).detach()
                D_perm = Discriminator(label_perm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, torch.zeros(img_1.size(0), dtype=torch.long, device=args.device)) + F.cross_entropy(D_perm,torch.ones(img_2.size(0), dtype=torch.long, device=args.device)))
                #print(label_tc_loss, D_tc_loss)
                optimizer_D.zero_grad()
                D_tc_loss.backward(retain_graph=True)
                optimizer.step()
                optimizer_D.step()
                label_list = label_list + list(D_label.detach().cpu()) +  list(label.detach().cpu())
            else : 
                loss0 = 0
                h_a0 = 0
            
            #stage 1
            optimizer_C.zero_grad()
            c_rec_loss, c_kl_loss, c_recon_img, mask_loss = model(img, gt, static, beta=args.c_beta, info= args.sup,stage=1)
            dag_param = model.dag.A # 4 x 4
            h_a1 = h_A(dag_param, dag_param.size()[0])
            loss1 = c_kl_loss + c_rec_loss + mask_loss + args.c_dag_w1*h_a1 + args.c_dag_w2 *h_a1*h_a1
            loss1.backward()
            optimizer_C.step()
            total_loss = total_loss +  loss0 + loss1
            total_DAG = total_DAG + (h_a0 + h_a1)/2
            total_c_rec += c_rec_loss.item()
            total_c_kl += c_kl_loss.item()


        if args.sup =='selfsup':
            label_list = torch.stack(label_list,dim=0)
            static= torch.stack((label_list.mean(dim=0), label_list.std(dim=0)), dim=1) # for self.scale
        else :
            static = None

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
                print(f'                    label recon: {total_l_rec/m}, label kl: {total_l_kl/m}, label_tc : {total_l_tc/m}')
                save_imgsets([img[0], c_recon_img[0], label_recon_img[0]], save_path)
                label_traverse(args, epoch, static, model,args.model_name, test_loader,pretrain=False)
            else :
                save_imgsets([img[0], c_recon_img[0]], save_path)
    return model, static
