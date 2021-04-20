import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from hvae.utils import Constants
from hvae.ops.manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero


def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)


# Обычный энкодер
class EncLinear(nn.Module):
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncLinear, self).__init__()
        self.manifold = manifold
        self.data_size = data_size

        # Слои
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])

        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


# Обычный декодер
class DecLinear(nn.Module):

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinear, self).__init__()
        self.data_size = data_size

        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])

        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)

# По сути обычный энкодер, у которого вконце экспоненциальное отображение (expmap0) для mu
class EncWrapped(nn.Module):

    # manifold - многообразие (PoincareBall)
    # non_lin - по умолчанию ReLU
    # prior_iso - isotropic prior, по умолчанию False
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncWrapped, self).__init__()
        self.manifold = manifold
        self.data_size = data_size

        # numpy.prod - произведение элементов массива
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)

        # coord_dim - размерность пространства
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


# Обычный декодер, только сначала применяется logmap0
class DecWrapped(nn.Module):

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrapped, self).__init__()
        self.data_size = data_size
        self.manifold = manifold

        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])

        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)

# ЗАМЕТКИ:
# Нужно самому создавать архитектуру нейросети (как Mnist или tabular в этом проекте)
# Примерно так:
# class myVAE_for_genomic(VAE):
#   def __init__(self, params): # В params будут распределения и сама архитектура сети
# Надеюсь трудностей не будет, дальше задаем optimizer (не знаю в модели или отдельно), loss_function, потом написать функцию train

# Сейчас непонятно: что делать с VAE и EncWrapped и DecWrapped. Нужно писать свой класс для нейросети для геномных данных?
# Или это все будет добавляться в VAE. В VAE есть метод forward и в энкодерах, декодерах тоже есть forward.

# Похоже действительно надо создавать свой класс VAE, т.к. переменная _pz_logvar не определна просто в VAE.

# Еще непонятно: как выбирается какой vae например для mnist. Мы указываем только когда в терминале --mnist. Но как
# Он приходит в mnist. Для этого повыводить в разных методах, чтобы понять


# Декодер. Пока не знаю как, попробовать:
# 1) первые слои линейные обычные, последний wrapped
# 2) все слои wrapped, надо посмотреть как в примерах, если есть
# Оказалось, что все нормально, отображаем из шара Пуанкаре в
# Евклидово пространство через логарифм, и остальные слои обычные

# Про eval(..) - выполняет строку, которая в аргументе. У нас например eval('Enc' + params.enc).
# Скорей всего, например, params.enc='Wrapped', будет строка 'EncWrapped' и eval вернет класс (объект) EncWrapped

# Попробовать, когда я экспериментировал в первый раз в колабе, повыводить принтом эти evalы
