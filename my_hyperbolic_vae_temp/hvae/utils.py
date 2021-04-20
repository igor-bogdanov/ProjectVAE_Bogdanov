import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
from torch.autograd import Variable, Function, grad

# Добавляет размерность тензору "слева"
def lexpand(A, *dimensions):
    return A.expand(tuple(dimensions) + A.shape)

# Добавляет размерность тензору справа
def rexpand(A, *dimensions):
    return A.view(A.shape + (1,) * len(dimensions)).expand(A.shape + tuple(dimensions))

##
def assert_no_nan(name, g):
    if(torch.isnan(g).any()):
        raise Exception('nans in {}'.format(name))

##
def assert_no_grad_nan(name, x):
    if(x.requires_grad):
        x.register_hook(lambda g: assert_no_nan(name, g))

# Classes
class Constants(object):
    eta = 1e-5
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi/2)

#torch.log(sinh(x))
#sinh(x) = (e^x - e^{-x}) / 2
def logsinh(x):
    return x + torch.log(1 - torch.exp(-2 * x)) - Constants.log2

# torch.log(cosh(x))
def logcosh(x):
    return x + torch.log(1 + torch.exp(-2 * x)) - Constants.log2

class Arccosh(Function):
    # https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x - 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


class Arcsinh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x + 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z

#Saves variables to the given filepath in a safe manner.
def save_vars(vs, filepath):
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)

#To load a saved model, simply use
#    `model.load_state_dict(torch.load('path-to-saved-model'))`.
def save_model(model, filepath):
    save_vars(model.state_dict(), filepath)

def log_mean_exp(value, dim=0, keepdim=False):
    return log_sum_exp(value, dim, keepdim) - math.log(value.size(dim))

def log_sum_exp(value, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))

def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))

def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()

def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    # elif len(params) == 3:
    #     return params[1]
    else:
        return params[0]