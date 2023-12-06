from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.ext import torch


# import torch

# we can merge all losses and select by passing args to `init` construct.
# for now following model initialization convention
# class WeightedMSELoss(torch.nn.MSELoss):
#
#     def __init__(self):
#         super(WeightedMSELoss, self).__init__()
#
#     # explain: the arguments passed to this forward must follow this order in the `train.py` script via gunpowder
#     def forward(self, lsds_prediction, lsds_target, lsds_weights, affs_prediction, affs_target, affs_weights):
#         # print(f"LSDs prediction {torch.unique(lsds_prediction)}")
#         # print(f"AFFs prediction {torch.unique(affs_prediction)}")
#
#         loss1 = super(WeightedMSELoss, self).forward(
#             lsds_prediction * lsds_weights,
#             lsds_target * lsds_weights)
#
#         loss2 = super(WeightedMSELoss, self).forward(
#             affs_prediction * affs_weights,
#             affs_target * affs_weights)
#
#         return loss1 + loss2


class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, lsds_prediction, lsds_target, lsds_weights, affs_prediction, affs_target, affs_weights):
        scaled_lsd = (lsds_weights * (lsds_prediction - lsds_target) ** 2)

        if len(torch.nonzero(scaled_lsd)) != 0:
            # only mean the loss where the values are non-zero
            mask_lsd = torch.masked_select(scaled_lsd, torch.gt(lsds_weights, 0))
            loss_lsd = torch.mean(mask_lsd)

        else:
            # will return 0 since all elements in scaled loss == 0
            loss_lsd = torch.mean(scaled_lsd)

        scaled_aff = (affs_weights * (affs_prediction - affs_target) ** 2)

        if len(torch.nonzero(scaled_aff)) != 0:

            mask_aff = torch.masked_select(scaled_aff, torch.gt(affs_weights, 0))
            loss_aff = torch.mean(mask_aff)

        else:
            loss_aff = torch.mean(scaled_aff)

        return loss_lsd + loss_aff


class WeightedMitoMSELoss(torch.nn.MSELoss):

    def __init__(self, task_lsd_wt=1, task_aff_wt=1, task_mito_wt=1):
        super(WeightedMitoMSELoss, self).__init__()
        self.task_lsd_wt = task_lsd_wt  # helps to tune each task in a multi-headed network
        self.task_aff_wt = task_aff_wt
        self.task_mito_wt = task_mito_wt

    def forward(self, lsds_prediction, lsds_target, lsds_weights, affs_prediction, affs_target, affs_weights,
                mito_prediction, mito_target, mito_weights):

        scaled_lsd = (lsds_weights * (lsds_prediction - lsds_target) ** 2)

        if len(torch.nonzero(scaled_lsd)) != 0:
            # only mean the loss where the values are non-zero
            mask_lsd = torch.masked_select(scaled_lsd, torch.gt(lsds_weights, 0))
            loss_lsd = torch.mean(mask_lsd)

        else:
            # will return 0 since all elements in scaled loss == 0
            loss_lsd = torch.mean(scaled_lsd)

        scaled_aff = (affs_weights * (affs_prediction - affs_target) ** 2)

        if len(torch.nonzero(scaled_aff)) != 0:

            mask_aff = torch.masked_select(scaled_aff, torch.gt(affs_weights, 0))
            loss_aff = torch.mean(mask_aff)

        else:
            loss_aff = torch.mean(scaled_aff)

        scaled_mito = (mito_weights * (mito_prediction - mito_target) ** 2)

        if len(torch.nonzero(scaled_mito)) != 0:

            mask_mito = torch.masked_select(scaled_mito, torch.gt(mito_weights, 0))
            loss_mito = torch.mean(mask_mito)

        else:
            loss_mito = torch.mean(scaled_mito)

        # this is a sum loss: all contributing equally
        # could be tuned 
        return self.task_lsd_wt * loss_lsd + self.task_aff_wt * loss_aff + self.task_mito_wt * loss_mito


class LSDWeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(LSDWeightedMSELoss, self).__init__()

    # explain: the arguments passed to this forward must follow this order in the `train.py` script via gunpowder
    def forward(self, lsds_prediction, lsds_target, lsds_weights):
        # sanity check: the lsds_predictions should not be always 0 or close to 0
        # print(torch.unique(lsds_prediction))
        # loss = super(LSDWeightedMSELoss, self).forward(
        #     lsds_prediction * lsds_weights,
        #     lsds_target * lsds_weights)

        scaled = (lsds_weights * (lsds_prediction - lsds_target) ** 2)

        if len(torch.nonzero(scaled)) != 0:
            # only mean the loss where the values are non-zero
            mask = torch.masked_select(scaled, torch.gt(lsds_weights, 0))
            loss = torch.mean(mask)

        else:
            # will return 0 since all elements in scaled loss == 0
            loss = torch.mean(scaled)

        return loss


class AFFWeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(AFFWeightedMSELoss, self).__init__()

    # explain: the arguments passed to this forward must follow this order in the `train.py` script via gunpowder
    def forward(self, affs_prediction, affs_target, affs_weights):

        # sometimes the targets can be totally blank images, since the labels do not exist for these raw slices maybe
        # we still optimise the loss for this case for now!!
        # we change this loss based on Arlo's latest `train_affinities.ipynb`
        scaled = (affs_weights * (affs_prediction - affs_target) ** 2)

        if len(torch.nonzero(scaled)) != 0:

            mask = torch.masked_select(scaled, torch.gt(affs_weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss


def initialize_loss(cfg):
    if cfg.TRAIN.MODEL_TYPE == "MTLSD":
        return WeightedMSELoss()
    elif cfg.TRAIN.MODEL_TYPE == "LSD":
        return LSDWeightedMSELoss()
    elif cfg.TRAIN.MODEL_TYPE == "AFF":
        return AFFWeightedMSELoss()
    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is None:
        # this is the trainable loss
        return AFFWeightedMSELoss()
    elif cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:
        # first train the LSD model
        return LSDWeightedMSELoss()


# if __name__ == '__main__':
#     initialize_loss(cfg)
