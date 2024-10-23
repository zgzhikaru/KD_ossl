from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PADLoss(nn.Module):
    """ Prime-Aware Distillation Loss
    code from """
    def __init__(self,
                 input_dim,
                 target_dim,
                 eps=1e-5
                 ):
        super(PADLoss, self).__init__()
        """
        self.pred_mean = nn.Sequential(
            nn.Linear(input_dim, target_dim), 
            nn.BatchNorm1d()
        )
        """
        self.log_variance = nn.Sequential(
            nn.Linear(input_dim, target_dim), 
            nn.BatchNorm1d(target_dim)
        )
        self.eps = eps

    def forward(self, input, target, return_logvar=False):
        # pool for dimension match

        assert len(input.shape) == 2 and len(target.shape) == 2, "Input needs to be a feature-vector"

        assert input.shape[1] == target.shape[1], "Input shape needs to match"

        pred_mean = input   #self.regressor(input)
        log_var_x = self.log_variance(input)
        #neg_log_prob = (pred_mean - target)**2/pred_var + log_var_x
        reg_term = torch.mean(log_var_x)

        pred_var = torch.exp(log_var_x) + self.eps
        neg_log_prob = torch.mean((pred_mean - target)**2/pred_var) + reg_term
        if return_logvar:
            return neg_log_prob, reg_term
        else:
            return neg_log_prob
