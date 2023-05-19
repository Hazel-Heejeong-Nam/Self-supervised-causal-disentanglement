from label_gen import BetaVAE_H
from utils import reconstruction_loss, kl_divergence
import tqdm

def pretrain(args):
    premodel = BetaVAE_H(args.z_dim, args.concept, args.z2_dim, args.pretrain_dec_type, enc_sep=args.pretrain_enc_sep).to(args.device)
    loader = 
    optim = optim.Adam(premodel.parameters(), lr=args.pre_lr, betas=(args.beta1, args.beta2))
    
    pbar = tqdm(total=iter)
    pbar.update(self.global_iter)
    
    for idx, x in enumerate(loader):
        pbar.update(1)

        x = x.to(args.device)
        x_recon, mu, logvar = premodel(x)
        recon_loss = reconstruction_loss(x, x_recon, args.decoder_dist)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)


        beta_vae_loss = recon_loss + args.beta*total_kld


        optim.zero_grad()
        beta_vae_loss.backward()
        optim.step()

        # epoch 으로 바꿨는데 visualize 어떻게 해야할지 모르겠어서 일단 없앰앰

    return premodel.state_dict()
