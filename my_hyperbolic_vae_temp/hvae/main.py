import os
import json
import math
import torch
from torch import optim
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import objectives
import models
from hvae.models.genomic_vae import GenomicVAE

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

#input_size = X.shape[1]
input_size=57
model = GenomicVAE(input_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = objectives.vae_objective

def train(epoch, batches_per_epoch=64, ):
    model.train() # from nn.Module
    b_loss, b_recon, b_kl = 0.0, 0.0, 0.0

    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)])
        data = Variable(data, requires_grad=False)
        optimizer.zero_grad()

        qz_x, px_z, lik, kl, loss = loss_function(model, data, 1, 1.0, components=True)

        loss.backward()
        optimizer.step()

        b_loss += loss.item()
        b_recon += -lik.mean(0).sum().item()
        b_kl += kl.sum(-1).mean(0).sum().item()

    if(epoch % 5 == 0):
      print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, b_loss / len(data)))
    print('====> Epoch: {} done!'.format(epoch))

if __name__ == '__main__':
    print('ok')
    print(type(model))
    #print(model.wtf_with_pa_params)
    print(model.pz_params)
    print('tra', model.training)

    for epoch in range(1, 31):
        train(epoch)

    print((model.parameters))
    print('p(z) params:')
    print(model.pz_params)
