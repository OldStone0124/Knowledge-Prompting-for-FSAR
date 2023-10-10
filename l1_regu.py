import torch.nn as nn
import torch
from ipdb import set_trace

class l1_regularizer():
    def __init__(self, opt, is_l1_regu_fn) -> None:
        self.l1_crit = nn.L1Loss(size_average=False)
        self.l1_factor = opt.l1_factor
        self.is_l1_regu_fn = is_l1_regu_fn

    def __call__(self, model):
        reg_loss = 0
        for name,param in model.named_parameters():
            if self.is_l1_regu_fn(name):
                # set_trace()
                target = torch.zeros_like(param)
                reg_loss += self.l1_crit(param, target)
        return reg_loss*self.l1_factor