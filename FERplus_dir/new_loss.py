import torch
import numpy as np
import torch.nn as nn

class New_Loss(nn.Module):
    def __init__(self):
        super(New_Loss, self).__init__()

    def forward(self, alphas_part_max, alphas_org):
        size = alphas_org.shape[0]
        loss_wt = 0.0
        for i in range(size):
            loss_wt += max(torch.Tensor([0]).cuda(), 0.1-(alphas_part_max[i]-alphas_org[i]))
        return loss_wt/size