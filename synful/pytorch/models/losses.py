from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.ext import torch


class SynPostMaskBCELoss(torch.nn.BCELoss):

    def __init__(self):
        super(SynPostMaskBCELoss, self).__init__(reduction='none')

    def forward(self, pred_syn_indicator, gt_syn_indicator, indicator_weight):
        # FROM b x 1 x d x h x w
        # TO  1 x d x h x w
        pred_syn_indicator = torch.squeeze(pred_syn_indicator, dim=0)
        # this is a matrix that needs to be scaled with weights
        scaled_post_mask = indicator_weight * super(SynPostMaskBCELoss, self).forward(pred_syn_indicator,
                                                                                      gt_syn_indicator.float())

        if len(torch.nonzero(scaled_post_mask)) != 0:
            # only mean the loss where the values are non-zero
            scaled_post_mask = torch.masked_select(scaled_post_mask, torch.gt(indicator_weight, 0))
            loss_post_mask = torch.mean(scaled_post_mask)
        else:
            # will return 0 since all elements in scaled loss == 0
            loss_post_mask = torch.mean(scaled_post_mask)

        return loss_post_mask


class SynPostMaskBCELogitLoss(torch.nn.BCEWithLogitsLoss):

    def __init__(self):
        super(SynPostMaskBCELogitLoss, self).__init__(reduction='none')

    def forward(self, pred_syn_indicator, gt_syn_indicator, indicator_weight):
        # FROM b x 1 x d x h x w
        # TO  1 x d x h x w
        pred_syn_indicator = torch.squeeze(pred_syn_indicator, dim=0)
        # this is a matrix that needs to be scaled with weights
        scaled_post_mask = indicator_weight * super(SynPostMaskBCELogitLoss, self).forward(pred_syn_indicator,
                                                                                           gt_syn_indicator.float())

        if len(torch.nonzero(scaled_post_mask)) != 0:
            # only mean the loss where the values are non-zero
            scaled_post_mask = torch.masked_select(scaled_post_mask, torch.gt(indicator_weight, 0))
            loss_post_mask = torch.mean(scaled_post_mask)
        else:
            # will return 0 since all elements in scaled loss == 0
            loss_post_mask = torch.mean(scaled_post_mask)

        return loss_post_mask


class SynPostMaskMSELoss(torch.nn.MSELoss):
    """
     MASK MSE Loss, but Mask must be a logit, i.e. without `Sigmoid` activation
    at the `conv3D` head.
    """

    def __init__(self):
        super(SynPostMaskMSELoss, self).__init__()

    def forward(self, pred_syn_indicator, gt_syn_indicator, indicator_weight):
        # FROM b x 1 x d x h x w
        # TO  1 x d x h x w
        pred_syn_indicator = torch.squeeze(pred_syn_indicator, dim=0)
        # this is a matrix that needs to be scaled with weights
        scaled_post_mask = (indicator_weight * (pred_syn_indicator - gt_syn_indicator.float()) ** 2)
        if len(torch.nonzero(scaled_post_mask)) != 0:
            # only mean the loss where the values are non-zero
            scaled_post_mask = torch.masked_select(scaled_post_mask, torch.gt(indicator_weight, 0))
            loss_post_mask = torch.mean(scaled_post_mask)
        else:
            # will return 0 since all elements in scaled loss == 0
            loss_post_mask = torch.mean(scaled_post_mask)

        return loss_post_mask


class SynPostVectorMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(SynPostVectorMSELoss, self).__init__()

    def forward(self, pred_syn_vector, gt_syn_vector, vector_mask):
        # FROM b x 1 x d x h x w
        # TO  1 x d x h x w
        pred_syn_vector = torch.squeeze(pred_syn_vector, dim=0)
        # this is a matrix that needs to be scaled with weights
        # scaled_post_vec = vector_mask * super(SynPostVectorMSELoss, self).forward(pred_syn_vector,
        #                                                                           gt_syn_vector.float())
        scaled_post_vec = (vector_mask * (pred_syn_vector - gt_syn_vector.float()) ** 2)

        if len(torch.nonzero(scaled_post_vec)) != 0:
            # only mean the loss where the values are non-zero
            scaled_post_vec = torch.masked_select(scaled_post_vec, torch.gt(vector_mask, 0))
            loss_post_vec = torch.mean(scaled_post_vec)
        else:
            # will return 0 since all elements in scaled loss == 0
            loss_post_vec = torch.mean(scaled_post_vec)

        return loss_post_vec


class WeightedSynLoss(torch.nn.MSELoss, torch.nn.BCEWithLogitsLoss):

    def __init__(self, m_scale=1., d_scale=1.):
        super(WeightedSynLoss, self).__init__()
        self.m_scale = m_scale
        self.d_scale = d_scale
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, pred_syn_indicator, gt_syn_indicator, indicator_weight, pred_syn_vector, gt_syn_vector,
                vector_mask):
        # FROM b x 1 x d x h x w
        # TO  1 x d x h x w
        # print(pred_syn_indicator.shape)
        # print(gt_syn_indicator.shape)
        # print(indicator_weight.shape)
        # print(gt_syn_vector.shape)
        # print(pred_syn_vector.shape)
        # print(vector_mask.shape)
        pred_syn_indicator = torch.squeeze(pred_syn_indicator, dim=0)
        # print(pred_syn_indicator.shape)
        # this is a matrix that needs to be scaled with weights
        scaled_post_mask = indicator_weight * self.bce_loss(pred_syn_indicator, gt_syn_indicator.float())
        # mse_loss - cannot multiply by vector_mask as shape:b=1xdxhxw
        scaled_post_vec = (vector_mask * (pred_syn_vector - gt_syn_vector.float()) ** 2)
        # print(torch.max(pred_syn_vector), torch.max(gt_syn_vector), torch.max(vector_mask))
        if len(torch.nonzero(scaled_post_vec)) != 0:
            # only mean the loss where the values are non-zero
            scaled_post_vec = torch.masked_select(scaled_post_vec, torch.gt(vector_mask, 0))
            # print(f"max/min scaled_post_vec loss {torch.max(scaled_post_vec)}, {torch.min(scaled_post_mask)}")
            loss_post_vec = torch.mean(scaled_post_vec)
        else:
            # will return 0 since all elements in scaled loss == 0
            loss_post_vec = torch.mean(scaled_post_vec)

        if len(torch.nonzero(scaled_post_mask)) != 0:
            # only mean the loss where the values are non-zero
            scaled_post_mask = torch.masked_select(scaled_post_mask, torch.gt(indicator_weight, 0))
            loss_post_mask = torch.mean(scaled_post_mask)
        else:
            # will return 0 since all elements in scaled loss == 0
            loss_post_mask = torch.mean(scaled_post_mask)

        return loss_post_mask * self.m_scale + loss_post_vec * self.d_scale


def initialize_loss(cfg):
    if cfg.TRAIN.MODEL_TYPE == "SynMT1":
        return WeightedSynLoss()
    elif cfg.TRAIN.MODEL_TYPE == "STMASK":
        # use this for now since the activation at the model head is `Sigmoid`
        return SynPostMaskBCELoss()
    elif cfg.TRAIN.MODEL_TYPE == "STVEC":
        return SynPostVectorMSELoss()
    elif cfg.TRAIN.MODEL_TYPE == "STMASK" and cfg.TRAIN.MASK_LOSS == "MSE":
        return SynPostMaskMSELoss()
    elif cfg.TRAIN.MODEL_TYPE == "STMASK" and cfg.TRAIN.MASK_LOSS == "BCELOGIT":
        return SynPostMaskBCELogitLoss()

# if __name__ == '__main__':
#     pass
