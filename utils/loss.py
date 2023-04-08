import torch
from torch.nn.modules.loss import _Loss
from torch.nn.utils import clip_grad_norm_
import numpy as np


def pairwise_distances(x):
    '''Taken from: https://discuss.pytorch.org/t/batched-pairwise-distance/39611'''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist[dist != dist] = 0  # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)

def dis_corr(z, data):
    z = z.reshape(z.shape[0], -1)
    data = data.reshape(data.shape[0], -1)
    a = pairwise_distances(z)
    b = pairwise_distances(data)
    a_centered = a - a.mean(dim=0).unsqueeze(1) - a.mean(dim=1) + a.mean()
    b_centered = b - b.mean(dim=0).unsqueeze(1) - b.mean(dim=1) + b.mean()
    dCOVab = torch.sqrt(torch.sum(a_centered * b_centered) / a.shape[1]**2)
    var_aa = torch.sqrt(torch.sum(a_centered * a_centered) / a.shape[1]**2)
    var_bb = torch.sqrt(torch.sum(b_centered * b_centered) / a.shape[1]**2)

    dCORab = dCOVab / torch.sqrt(var_aa * var_bb)
    return dCORab


