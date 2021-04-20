import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader

import math
from sklearn.model_selection._split import _validate_shuffle_split
from .vae import VAE
from hvae.vis import array_plot

from hvae.manifolds.poincare import PoincareBall

from hvae.distributions.wrapped_normal import WrappedNormal
from torch.distributions import Normal
from hvae import manifolds
from hvae.models.architectures import EncLinear, DecLinear, EncWrapped, DecWrapped
#from hvae.datasets import CSVDataset
# Скорей всего добавить класс CSVDataset из datasets.py


class GenomicVAE(VAE):
    # Что нужно указать: latent_dim
    # c? Кривизна
    # prior_dist
    # posterior_dist
    # likelihood
    # data_size
    # non_lin (ReLU)
    # num_hidden_layers
    # hidden_dim
    # prior_iso
    def __init__(self, data_size):
        curv = 0.7
        c = nn.Parameter(curv * torch.ones(1), requires_grad=False)

        # PoincareBall;    latent_dim=2
        # manifold = PoincareBall(params.latent_dim, c)
        latent_dim = 2
        manifold = PoincareBall(latent_dim, c)
        #manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)

        # Сккорей всего установить самому, не через params
        prior = 'WrappedNormal'
        posterior = 'WrappedNormal'


        super(GenomicVAE, self).__init__(
            eval(prior),         # prior dist
            eval(posterior),     # posterior dist
            dist.Normal,                # likelihood dist

            EncWrapped(manifold, data_size, nn.ReLU(), num_hidden_layers=1, hidden_dim=200, prior_iso=True),
            DecWrapped(manifold, data_size, nn.ReLU(), num_hidden_layers=1, hidden_dim=200)
            #params
        )

        self.manifold = manifold
        self._pz_mu = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

    @property
    def pz_params(self):
        #print(self)
        #return
        #return self._pz_mu, self.manifold
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(1.), self.manifold
        #return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    def get_z(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        return zs


    # Не нужно
    #def generate(self, runP):
    #    N, K = 10, 1
    #    _, _, samples

    # Не нужно
    #def reconstruct(self, data):