from funlib.learn.torch.models import UNet, ConvPass
from torchinfo import summary
from gunpowder.ext import torch
import logging

# import the same logger
logger = logging.getLogger(__name__)


class SynPostMaskModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            pad_conv='valid',
            num_fmaps_out=12,
            nhood=1,  # this is just mask hence out_channel=1
            use_2d=False,
            activation: (str | None) = 'Sigmoid'
    ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            num_fmaps_out=num_fmaps_out,
            padding=pad_conv
        )

        if use_2d:
            # mask shape: 1 x D x H X W
            self.post_syn_mask_head = ConvPass(num_fmaps_out, nhood, [[1, 1]], activation=activation)

        else:
            # mask shape: 1 x D x H X W
            self.post_syn_mask_head = ConvPass(num_fmaps_out, nhood, [[1, 1, 1]], activation=activation)

    def forward(self, x):
        z = self.unet(x)

        # if ``activation=None`` above, then ``post_syn_mask`` a logit.
        # Ensure you do ``Sigmoid(post_syn_mask)`` before using ``BCELoss()``.
        # Else, ``BCELossWithLogits()`` should do it by default.
        # Shape: b x c x d x h x w
        post_syn_mask = self.post_syn_mask_head(z)

        return post_syn_mask


class SynPostVectorModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            pad_conv='valid',
            num_fmaps_out=12,
            nhood=3,  # this is direction vector, hence out_channel=3 for 3D
            use_2d=False,
            activation='Sigmoid',
            cfg=None
    ):
        super().__init__()

        if cfg is not None:
            self.cfg = cfg

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            num_fmaps_out=num_fmaps_out,
            padding=pad_conv
        )

        if use_2d:
            # mask shape: 1 x D x H X W
            self.post_syn_vec_head = ConvPass(num_fmaps_out, nhood, [[1, 1]], activation=activation)

        else:
            # mask shape: 1 x D x H X W
            self.post_syn_vec_head = ConvPass(num_fmaps_out, nhood, [[1, 1, 1]], activation=activation)

    def forward(self, x):
        z = self.unet(x)

        # Shape: b x c x d x h x w
        post_syn_vec = self.post_syn_vec_head(z)

        return post_syn_vec


class SynMT1DoubleHeadedModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            pad_conv='valid',
            num_fmaps_out=12,
            nhood=3,  # this is direction vector, hence out_channel=3 for 3D
            use_2d=False,
            activation='Sigmoid'
    ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            num_fmaps_out=num_fmaps_out,
            padding=pad_conv
        )

        if use_2d:
            # mask shape: b x 1 x D x H X W
            # vec shape: b x 2 x D x H x W
            self.post_syn_vec_head = ConvPass(num_fmaps_out, 2, [[1, 1]], activation=activation)
            self.post_syn_mask_head = ConvPass(num_fmaps_out, 1, [[1, 1]], activation=activation)

        else:
            # mask shape: b x 1 x D x H X W
            # vec shape: b x 3 x D x H X W
            self.post_syn_vec_head = ConvPass(num_fmaps_out, 3, [[1, 1, 1]], activation=activation)
            self.post_syn_mask_head = ConvPass(num_fmaps_out, 1, [[1, 1, 1]], activation=activation)

    def forward(self, x):
        z = self.unet(x)

        # Shape: b x c x d x h x w
        post_syn_vec = self.post_syn_vec_head(z)
        post_syn_mask = self.post_syn_mask_head(z)

        return post_syn_mask, post_syn_vec


class CalculateModelSummary:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def calculate_output_shape(self, input_shape=None):
        try:
            if input_shape is None:
                input_shape = self.cfg.MODEL.INPUT_SHAPE
                if self.cfg.DATA.DIM_2D:
                    input_shape = self.cfg.MODEL.INPUT_SHAPE_2D
                # expects B X C X (D-OPTIONAL) X H X W
                image_dim = (1, 1,) + input_shape

            else:
                image_dim = input_shape
            output_shape = self.model(torch.rand(*(image_dim)))
            if isinstance(output_shape, tuple):
                # assuming all models take in same input shape and spit out same output shape, grab the first one
                output_shape = output_shape[0].data.shape
            else:
                output_shape = output_shape.data.shape

            logger.info(f"Expected output shape pre-calculated from input shape {output_shape}")

            if self.cfg.SYSTEM.VERBOSE:
                # Sending the model to cuda is solution to: RuntimeError:
                # Input type (torch.cuda.FloatTensor)
                # and weight type (torch.FloatTensor) should be the same
                logger.info(f"Model Summary stats with expected output shape"
                            f"{summary(self.model, image_dim, device='cpu')}")

                # print to sys.output screen
                # print(summary(self.model.to('cuda'), (1,) + input_shape))

        except Exception as e:
            raise RuntimeError(e)

        return output_shape


# Todo: Add the MT2 network with independent upsampling paths
def initialize_model(cfg):
    if cfg.TRAIN.MODEL_TYPE == "STMASK":
        return SynPostMaskModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            # ain't technically neighborhood, maintain variable for consistency with LSD
            nhood=1,
            use_2d=cfg.DATA.DIM_2D)

    elif cfg.TRAIN.MODEL_TYPE == "STVEC":
        return SynPostVectorModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            pad_conv=cfg.MODEL.PAD_CONV,
            nhood=3 if not cfg.DATA.DIM_2D else 2,
            use_2d=cfg.DATA.DIM_2D,
            cfg=cfg)

    elif cfg.TRAIN.MODEL_TYPE == "SynMT1":
        return SynMT1DoubleHeadedModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            pad_conv=cfg.MODEL.PAD_CONV,
            nhood=3 if not cfg.DATA.DIM_2D else 2,
            use_2d=cfg.DATA.DIM_2D)
    elif cfg.TRAIN.MODEL_TYPE == "STMASK" and cfg.TRAIN.MASK_LOSS in ["MSE", "BCELOGIT"]:
        return SynPostMaskModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            # ain't technically neighborhood, maintain variable for consistency with LSD
            nhood=1,
            use_2d=cfg.DATA.DIM_2D,
            activation=None)  # a logit is return, loss must be changed
