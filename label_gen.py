from utils import Encoder, Decoder_DAG
from torch import nn
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10,concept=4, z2_dim=4,dec_type = 'separate', nc=3, enc_sep:bool=False):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        ###
        self.enc_sep = enc_sep
        self.concept = concept
        self.z2_dim = z2_dim
        self.dec_type = dec_type
        ###
        self.nc = nc
        self.encoder = Encoder(z_dim=self.z_dim, concept = self.concept, z2_dim = self.z2_dim, channel=3, y_dim=4, separate=self.enc_sep)
        self.decoder = Decoder_DAG(z_dim = self.z_dim, concept = self.concept, z2_dim = self.z2_dim, channel = 3, y_dim=0)
        #self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             kaiming_init(m)

    def forward(self, x):
        mu, logvar = self.encoder(x) # mu : bs x 4 logvar : bs x 4
        # mu = distributions[:, :self.z_dim]
        # logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar) # bs x 4
        if self.dec_type == 'separate':
            x_recon= self.decoder.decode_sep(z)
        elif self.dec_type == 'integrate':
            x_recon= self.decoder.decode(z)
        else :
            NotImplementedError('Invalid dypte of decoder')
        x_recon = x_recon.view(x_recon.shape[0],3,96,96)
        return x_recon, mu, logvar