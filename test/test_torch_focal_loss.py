# coding: utf-8
import torch
import torch.nn as nn

from feelfree.torch.focal_loss import MultiFocalLoss

def test_focal_loss():
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    fl_loss = MultiFocalLoss(3, alpha=[0.4, 0.3, 0.3], gamma=2, balance_index=-1, smooth=None, size_average=True)
    input = torch.randn(3, 3, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(3)
    ce_out = ce_loss(input, target)
    fl_out = fl_loss(input, target)

    assert fl_out < ce_out

