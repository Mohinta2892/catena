from funlib.learn.torch.models import UNet, ConvPass
from torchinfo import summary
from gunpowder.ext import torch
import logging

# import the same logger
logger = logging.getLogger(__name__)


class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out=12,
            pad_conv='valid',  # default valid convolutions, which result in smaller output shape than input
            nhood=3,
            use_2d=False,
            lsds=10  # num of local shape descriptors; 10 for 3D/ 6 for 2D
    ):
        super().__init__()

        self.lsds_features = lsds
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
            self.lsds_features = 6
            self.lsd_head = ConvPass(num_fmaps_out, self.lsds_features, [[1, 1]],
                                     activation='Sigmoid')  # LSD feature maps: 6 x D x H X W
            # Affinity default: 2 x D x H X W; could also 6 x D x H x W if LR-affs
            self.aff_head = ConvPass(num_fmaps_out, nhood, [[1, 1]], activation='Sigmoid')

        else:
            # LSD feature maps: 10 x D x H X W
            self.lsd_head = ConvPass(num_fmaps_out, self.lsds_features, [[1, 1, 1]], activation='Sigmoid')
            # Affinity default: 3 x D x H X W; could also >3 x D x H x W if LR-affs
            self.aff_head = ConvPass(num_fmaps_out, nhood, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, x):
        z = self.unet(x)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        # print(f"affinity out size {affs.shape}")
        # print(f"affinity out size {lsds.shape}")

        return lsds, affs


class MtlsdMitoModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out=12,
            pad_conv='valid',  # default valid convolutions, which result in smaller output shape than input
            nhood=3,
            use_2d=False,
            lsds=10  # num of local shape descriptors; 10 for 3D/ 6 for 2D
    ):
        super().__init__()

        self.lsds_features = lsds
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
            self.lsds_features = 6
            self.lsd_head = ConvPass(num_fmaps_out, self.lsds_features, [[1, 1]],
                                     activation='Sigmoid')  # LSD feature maps: 6 x D x H X W
            # Affinity default: 2 x D x H X W; could also 6 x D x H x W if LR-affs
            self.aff_head = ConvPass(num_fmaps_out, nhood, [[1, 1]], activation='Sigmoid')
            # Mito affinities: 2 x D x H x W
            self.mito_head = ConvPass(num_fmaps_out, nhood, [[1, 1]], activation='Sigmoid')

        else:
            # LSD feature maps: 10 x D x H X W
            self.lsd_head = ConvPass(num_fmaps_out, self.lsds_features, [[1, 1, 1]], activation='Sigmoid')
            # Affinity default: 3 x D x H X W; could also >3 x D x H x W if LR-affs
            self.aff_head = ConvPass(num_fmaps_out, nhood, [[1, 1, 1]], activation='Sigmoid')
            # Mito affinities: 3 x D x H x W
            self.mito_head = ConvPass(num_fmaps_out, nhood, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, x):
        z = self.unet(x)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)
        mito = self.mito_head(z)

        # print(f"affinity out size {affs.shape}")
        # print(f"affinity out size {lsds.shape}")

        return lsds, affs, mito


class LsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out=12,
            pad_conv='valid',
            nhood=3,  # does not affect the output dims
            use_2d=False,
            lsds=10,  # num of local shape descriptors; 10 for 3D/ 6 for 2D,
            cfg=None
    ):
        super().__init__()
        self.lsds_features = lsds
        if cfg is not None:
            self.cfg = cfg
        self.use_2d = use_2d

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
            self.lsds_features = 6  # overwrite here for now
            self.lsd_head = ConvPass(num_fmaps_out, self.lsds_features, [[1, 1]],
                                     activation='Sigmoid')  # LSD feature maps: 6 x D x H X W

        else:
            # LSD feature maps: 10 x D x H X W
            self.lsd_head = ConvPass(num_fmaps_out, self.lsds_features, [[1, 1, 1]], activation='Sigmoid')

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        if self.use_2d:
            x_target_size = x.size()[:-2] + shape
        else:
            x_target_size = x.size()[:-3] + shape

        offset = tuple(
            (a - b) // 2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        z = self.unet(x)
        lsds = self.lsd_head(z)

        if self.cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"]:
            # we have to crop the lsds to some intermediate shape if lsd_shape > intermediate shape
            if lsds.shape[-3:] > self.cfg.MODEL.INTERMEDIATE_SHAPE:
                lsds = self.crop(lsds, self.cfg.MODEL.INTERMEDIATE_SHAPE)
        if self.cfg.TRAIN.MODEL_TYPE == "ACRLSD" and self.cfg.TRAIN.LSD_EPOCHS is None:
            # in predict mode the raw must be concatenated with the lsds in the channel dim
            # x shape = B x C x (D- if 3D) x H x W
            if x.shape[-3:] > self.cfg.MODEL.INTERMEDIATE_SHAPE:
                x_crop = self.crop(x, self.cfg.MODEL.INTERMEDIATE_SHAPE)
            else:
                x_crop = x
            lsds = torch.cat((x_crop, lsds), dim=1)

        return lsds


class AffModel(torch.nn.Module):

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
            nhood=3,
            use_2d=False
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
            # Affinity default: 2 x D x H X W; could also 6 x D x H x W if LR-affs
            self.aff_head = ConvPass(num_fmaps_out, nhood, [[1, 1]], activation='Sigmoid')

        else:
            # Affinity default: 3 x D x H X W; could also 6 x D x H x W if LongRange-affinities
            self.aff_head = ConvPass(num_fmaps_out, nhood, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, x):
        z = self.unet(x)
        affs = self.aff_head(z)

        return affs


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


def initialize_model(cfg):
    if cfg.TRAIN.MODEL_TYPE == "MTLSD":
        return MtlsdModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
            use_2d=cfg.DATA.DIM_2D)

    elif cfg.TRAIN.MODEL_TYPE == "LSD":
        return LsdModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            pad_conv=cfg.MODEL.PAD_CONV,
            nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
            use_2d=cfg.DATA.DIM_2D,
            cfg=cfg)

    elif cfg.TRAIN.MODEL_TYPE == "AFF":
        return AffModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            num_fmaps=cfg.MODEL.NUM_FMAPS,
            fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
            downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
            kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
            kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
            num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
            pad_conv=cfg.MODEL.PAD_CONV,
            nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
            use_2d=cfg.DATA.DIM_2D)

    elif cfg.TRAIN.MODEL_TYPE == "ACLSD":
        # we need to return two models for auto-context setup,
        # such that AFFModel gets trained with the predictions from LSDModel
        return (
            LsdModel(
                in_channels=cfg.MODEL.IN_CHANNELS,
                num_fmaps=cfg.MODEL.NUM_FMAPS,
                fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
                downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
                kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
                kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
                num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
                pad_conv=cfg.MODEL.PAD_CONV,
                nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
                use_2d=cfg.DATA.DIM_2D, cfg=cfg),

            AffModel(
                in_channels=cfg.MODEL.LSDS,
                num_fmaps=cfg.MODEL.NUM_FMAPS,
                fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
                downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
                kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
                kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
                num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
                pad_conv=cfg.MODEL.PAD_CONV,
                nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
                use_2d=cfg.DATA.DIM_2D)
        )
    elif cfg.TRAIN.MODEL_TYPE == "ACRLSD":
        # we need to return two models for auto-context setup,
        # such that AFFModel gets trained with the predictions from LSDModel
        return (
            LsdModel(
                in_channels=cfg.MODEL.IN_CHANNELS,
                num_fmaps=cfg.MODEL.NUM_FMAPS,
                fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
                downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
                kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
                kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
                num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
                pad_conv=cfg.MODEL.PAD_CONV,
                nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
                use_2d=cfg.DATA.DIM_2D, cfg=cfg),

            AffModel(
                in_channels=cfg.MODEL.LSDS + 1,  # an additional channel for concatenated raw
                num_fmaps=cfg.MODEL.NUM_FMAPS,
                fmap_inc_factor=cfg.MODEL.FMAP_INC_FACTOR,
                downsample_factors=cfg.MODEL.DOWNSAMPLE_FACTORS if not cfg.DATA.DIM_2D else cfg.MODEL.DOWNSAMPLE_FACTORS_2D,
                kernel_size_down=cfg.MODEL.KERNEL_SIZE_DOWN if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_DOWN_2D,
                kernel_size_up=cfg.MODEL.KERNEL_SIZE_UP if not cfg.DATA.DIM_2D else cfg.MODEL.KERNEL_SIZE_UP_2D,
                num_fmaps_out=cfg.MODEL.NUM_FMAPS_OUT,
                pad_conv=cfg.MODEL.PAD_CONV,
                nhood=len(cfg.TRAIN.NEIGHBORHOOD) if not cfg.DATA.DIM_2D else len(cfg.TRAIN.NEIGHBORHOOD_2D),
                use_2d=cfg.DATA.DIM_2D)
        )

# if __name__ == "__main__":
#     initialize_model(cfg)
