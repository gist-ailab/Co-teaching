import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = torch.argsort(loss_1)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = torch.argsort(loss_2)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # print(noise_or_not, ind)
    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2


# Loss functions
def loss_multiteaching(y_, t, forget_rate, ind, noise_or_not):
    loss_ = dict()
    loss_mean = dict()
    pure_ratio = dict()

    for n in range(len(y_)):
        loss_[n] = F.cross_entropy(y_[n], t, reduce = False)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_[n]))

    # print(noise_or_not, ind)
    for n in range(len(y_)):
        # Calculate Loss for n-th model
        loss_sum = torch.zeros_like(loss_[n])
        other_idx = list(range(len(y_)))
        other_idx.pop(n)
        for j in other_idx:
            loss_sum += loss_[j]
        ind_sorted = torch.argsort(loss_sum)
        loss_sorted = loss_[n][ind_sorted]

        pure_ratio_ = torch.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

        ind_update=ind_sorted[:num_remember]

        # exchange
        loss_update = F.cross_entropy(y_[n][ind_update], t[ind_update])
        loss_mean[n] = loss_update / num_remember
        pure_ratio[n] = pure_ratio_

    return loss_mean, pure_ratio