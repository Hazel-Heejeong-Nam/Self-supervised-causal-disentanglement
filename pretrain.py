
from utils import reconstruction_loss, kl_divergence
import tqdm

def pretrain_labelgen(args, model, loader, optimizer):
    #pbar = tqdm(total=len(loader)) # 임의임
    
    for epoch in range(args.pretrain_epoch):
        for  idx, (x, gt ) in enumerate(loader):
            #pbar.update(1)

            x = x.to(args.device)
            x_recon, label, labelvar = model(x)
            recon_loss = reconstruction_loss(x, x_recon, args.decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(label, labelvar)
            # no additional loss since pretraining

            beta_vae_loss = recon_loss + args.beta*total_kld


            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()

        # epoch 으로 바꿨는데 visualize 어떻게 해야할지 모르겠어서 일단 없앰앰

    return model
