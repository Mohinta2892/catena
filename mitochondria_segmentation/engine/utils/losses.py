import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        sig00 = nn.Sigmoid()
        inputs = sig00(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceCE(nn.Module):
    def __init__(self):
        super(DiceCE, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_loss_fn = DiceLoss()

    def forward(self, inputs, targets, loss_weights, smooth=1):
        ce_loss_instance = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weights).to(inputs.device))
        CE = ce_loss_instance(inputs, targets)

        DICE = self.dice_loss_fn(inputs, targets)

        return (CE + DICE)
