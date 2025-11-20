import torch
import numpy as np
from funlib.learn.torch.models import UNet, ConvPass
import torch.nn.functional as F


class MtSynfulModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            num_fmaps_out=12):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            # kernel_size_down=kernel_size_down,
            # kernel_size_up=kernel_size_up,
            # num_fmaps_out=num_fmaps_out
        )

        self.mask_head = ConvPass(num_fmaps, 1, [[1, 1, 1]], activation=None)
        self.dir_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation=None)

    def forward(self, input):
        z = self.unet(input)
        print(f"z-shape {z.size()}")
        locs = self.mask_head(z)
        dirs = self.dir_head(z)

        return locs, dirs


class WeightedMTSynfulLoss(torch.nn.MSELoss, torch.nn.BCEWithLogitsLoss):
    def __init__(self, m_loss_scale, d_loss_scale):
        super(WeightedMTSynfulLoss, self).__init__()
        self.m_loss_scale = m_loss_scale
        self.d_loss_scale = d_loss_scale
        self.obj_mseloss = torch.nn.MSELoss(reduction='none')
        self.obj_bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def partner_vectors_loss_mask(self, gt_partner_vectors, pred_partner_vectors, vectors_mask):
        # print("CHECKING FOR NANS IN VECTORS LOSS--------->>>")
        # print(torch.sum(torch.isnan(gt_partner_vectors)), torch.sum(torch.isnan(pred_partner_vectors)),
        #       torch.sum(torch.isnan(vectors_mask)))
        loss = self.obj_mseloss(torch.squeeze(pred_partner_vectors),
                                gt_partner_vectors.float())  # .float() is equivalent to float32()
        # vectors_mask = vectors_mask.to(torch.bool)
        # print(
        # f"LOSS VECTOR LOSS--> {loss.shape}, vectors sum --> {torch.sum(vectors_mask)}, vectors sum --> {vectors_mask.shape}")
        print(
            f"LOSS VECTOR LOSS before vectors masking--> {torch.sum(loss)}, vectors sum --> {torch.sum(vectors_mask)}")

        loss = torch.sum(loss * vectors_mask.view((1,) + vectors_mask.shape))
        # assert torch.sum(vectors_mask) != 0
        print(f"LOSS VECTOR LOSS after vectors masking--> {loss}, vectors sum --> {torch.sum(vectors_mask)}")
        if torch.sum(vectors_mask) > 0:
            loss = loss / torch.sum(vectors_mask)
        else:
            loss = loss * torch.sum(vectors_mask)
        return loss

    def syn_indicator_loss_weighted(self, gt_syn_indicator, pred_syn_indicator, indicator_weight):
        print("CHECKING FOR NANS IN SYN INDICATOR LOSS--------->>>")
        print(torch.sum(torch.isnan(gt_syn_indicator)), torch.sum(torch.isnan(pred_syn_indicator)),
              torch.sum(torch.isnan(indicator_weight)))
        pred_syn_indicator = torch.squeeze(pred_syn_indicator)
        # print(gt_syn_indicator.shape, pred_syn_indicator.shape, indicator_weight.shape)
        loss = self.obj_bceloss(pred_syn_indicator, gt_syn_indicator.float())
        loss = torch.sum(loss * indicator_weight)
        print(f"LOSS SYN INDICATOR LOSS--> {loss}, indicator weight sum --> {torch.sum(indicator_weight)}")
        loss = loss / torch.sum(indicator_weight)
        return loss

    def forward(self, gt_partner_vectors, pred_partner_vectors, vectors_mask, gt_syn_indicator,
                pred_syn_indicator, indicator_weight):
        partner_vectors_loss = self.partner_vectors_loss_mask(gt_partner_vectors, pred_partner_vectors, vectors_mask)
        syn_indicator_loss = self.syn_indicator_loss_weighted(gt_syn_indicator, pred_syn_indicator, indicator_weight)
        loss = self.m_loss_scale * syn_indicator_loss + self.d_loss_scale * partner_vectors_loss
        return loss


class STMaskSynfulModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            num_fmaps_out=12):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            # kernel_size_down=kernel_size_down,
            # kernel_size_up=kernel_size_up,
            # num_fmaps_out=num_fmaps_out
        )

        self.mask_head = ConvPass(num_fmaps, 1, [[1, 1, 1]], activation=None)

    def forward(self, input):
        # print("input shape --->", input.shape)
        z = self.unet(input)
        logits = self.mask_head(z)
        predictions = torch.sigmoid(logits)

        # print("mask --->", torch.unique(locs))

        return logits, predictions


class STMaskSynfulLoss(torch.nn.Module):
    def __init__(self):
        super(STMaskSynfulLoss, self).__init__()
        self.obj_bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_syn_indicator, gt_syn_indicator, indicator_weight):
        # FROM b x c x d x h x w  TO  c x d x h x w
        # pred_syn_indicator = logits from the above forward without Sigmoid since we use LogitsLoss
        pred_syn_indicator = torch.squeeze(pred_syn_indicator)
        # print("gt_syn_indicator -->", torch.unique(gt_syn_indicator))
        # print("pred_syn_indicator -->", torch.unique(pred_syn_indicator))
        # # print("indicator_weight -->", torch.unique(indicator_weight))
        loss = self.obj_bceloss(pred_syn_indicator, gt_syn_indicator.float())
        loss = torch.sum(loss * indicator_weight)
        # # # print(f"LOSS SYN INDICATOR LOSS--> {loss}, indicator weight sum --> {torch.sum(indicator_weight)}")
        loss = loss / torch.sum(indicator_weight)
        return loss
