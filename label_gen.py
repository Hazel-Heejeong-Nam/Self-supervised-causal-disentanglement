from utils import Encoder_share, Decoder_DAG, Encoder_label
from torch import nn
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class label_beta(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10,concept=4, z2_dim=4, nc=4):
        super(label_beta, self).__init__()
        self.z_dim = z_dim
        ###
        self.concept = concept
        self.z2_dim = z2_dim
        ###
        self.nc = nc
        self.encoder_share = Encoder_share(z_dim=self.z_dim, channel=self.nc)
        self.label_encoder = Encoder_label(z_dim= self.z_dim, concept=self.concept)
        self.decoder = Decoder_DAG(z_dim = self.z_dim, concept = self.concept, z2_dim = self.z2_dim, channel = self.nc, y_dim=0)
        #self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             kaiming_init(m)

    def forward(self, x):
        mu, logvar = self.encoder_share(x) # mu : bs x 4 logvar : bs x 4
        label, labelvar = self.label_encoder(mu)
        z = reparametrize(label, labelvar) # bs x concept
        
        # missing param 있대
        x_recon= self.decoder.decode_label(z)
        x_recon = x_recon.view(x_recon.shape[0],self.nc,96,96)
        return x_recon, label, labelvar