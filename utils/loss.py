''' Mahmoud, dcor from NoPeek code '''

'''

def pairwise_dist(A):
    # Taken frmo https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    #A = tf_print(A, [tf.reduce_sum(A)], message="A is")
    r = tf.reduce_sum(A*A, 1)
    #r = tf_print(r, [tf.reduce_sum(r)], message="r is")
    r = tf.reshape(r, [-1, 1])
    D = tf.maximum(r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r), 1e-7)
    D = tf.sqrt(D)
    return D

def dist_corr(X, Y):
    n = tf.cast(tf.shape(X)[0], tf.float32)
    a = pairwise_dist(X)
    b = pairwise_dist(Y)
    A = a - tf.reduce_mean(a, axis=1) - tf.expand_dims(tf.reduce_mean(a, axis=0), axis=1) + tf.reduce_mean(a)
    B = b - tf.reduce_mean(b, axis=1) - tf.expand_dims(tf.reduce_mean(b, axis=0), axis=1) + tf.reduce_mean(b)
    dCovXY = tf.sqrt(tf.reduce_sum(A*B) / (n ** 2))
    dVarXX = tf.sqrt(tf.reduce_sum(A*A) / (n ** 2))
    dVarYY = tf.sqrt(tf.reduce_sum(B*B) / (n ** 2))
    
    dCorXY = dCovXY / tf.sqrt(dVarXX * dVarYY)
    return dCorXY

def custom_loss1(y_true,y_pred):
    #y_pred = tf_print(y_pred, [tf.reduce_sum(y_pred)], message="y_pred is")
    dcor = dist_corr(y_true,y_pred)
    #dcor = tf_print(dcor, [tf.reduce_sum(dcor)], message="dcor is")
    return dcor

def custom_loss2(y_true,y_pred):
    #y_pred = tf_print(y_pred, [tf.reduce_sum(y_pred)], message="y_pred is")
    recon_loss = losses.categorical_crossentropy(y_true, y_pred)
    return recon_loss

'''


import torch
import torch.nn as nn
#import tensorflow as tf


# code from https://github.com/gkasieczka/DisCo/blob/master/Disco.py

def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    normedweight = 1
    var_1 = torch.reshape(var_1, [-1, 1])
    # var_2.detach().numpy()
    var_2 = torch.reshape(var_2, [-1, 1])
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    # bmat = bmat.detach().numpy()
    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr



''' new Dcor code '''

# from algos.simba_algo import SimbaDefence
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



def PatchShuffle(x):
    """
    https://github.com/dixiyao/Patch-Shuffling-Transformer/blob/acd17e543b04cac869806eae58d56f632ce23bac/utilsenc.py#L24
    """
    for bs in range(x.shape[0]):
        # random permutation
        x[bs] = x[bs][torch.randperm(x.shape[1]),:]
    return x