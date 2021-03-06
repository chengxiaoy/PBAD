from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
import numpy as np
from math import floor
from config import Config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):

        if Config.USE_GAUSSIAN:
            focal_v2 = FocalLoss_v2()
            return focal_v2(inputs, targets)

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)

        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.log_vars = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.log_vars_regr = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32), requires_grad=True)

    def forward(self, cls, regr):
        # return 1 / 2 * torch.exp(-self.log_vars) * cls + torch.exp(
        #     -self.log_vars_regr) * regr + 1 / 2 * self.log_vars + 1 / 2 * self.log_vars_regr
        return 1 / 2 * torch.exp(-self.log_vars) * cls + torch.exp(
            -self.log_vars_regr) * regr + torch.exp(1 / 2 * self.log_vars) + torch.exp(1 / 2 * self.log_vars_regr)


class FocalLoss_v2(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss_v2, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
