# import logging
# import random
#
# from gunpowder.nodes.batch_filter import BatchFilter
# from gunpowder.profiling import Timing
#
# logger = logging.getLogger(__name__)
# #
#
# class MaskReject(BatchFilter):
#     """Reject batches based on the masked-in vs. masked-out ratio.
#     Author: William Patton, copied from ContextNet, HHMI Janelia, USA
#     Link: `https://github.com/pattonw/contextnet/blob/main/contextnet/gp/reject_if_empty.py`
#
#     Args:
#
#         gt (``np.ndarray``):
#
#             The gt array to use
#
#         p (``float``, optional):
#
#             The probability that we reject until gt is nonempty
#
#         background (``int``, optional):
#
#             Value that denotes the background in the image/volume
#     """
#
#     def __init__(self, mask, raw, p=0.5, background=0):
#
#         self.mask = mask
#         self.raw = raw
#         self.p = p
#         self.background = background
#
#     def setup(self):
#         upstream_providers = self.get_upstream_providers()
#         assert len(upstream_providers) == 1, "Only 1 upstream provider supported"
#         self.upstream_provider = upstream_providers[0]
#
#     def provide(self, request):
#         random.seed(request.random_seed)
#
#         report_next_timeout = 10
#         num_rejected = 0
#
#         timing = Timing(self)
#         timing.start()
#
#         batch = self.upstream_provider.request_batch(request)
#         raw_roi = batch.arrays[self.raw].roi
#         # mask_data = get_mask_data_in_roi(mask=mask[, roi=raw_roi, target_voxel_size=raw_roi.voxel_size)
#
#
#         # assert self.gt in request, f"Cannot reject on {self.gt} if its not requested"
#
#         have_good_batch = random.random() < self.p
#         while True:
#             batch = self.upstream_provider.request_batch(request)
#             mask_data = batch.arrays[self.mask].data
#             raw_data = batch.arrays[self.raw].data
#
#             # empty = (gt_data.min() == self.background) and (
#             #         gt_data.max() == self.background
#             # )
#             # print(gt_data.min(), gt_data.max(), have_good_batch)
#
#             empty = mask_data.sum() == 0
#
#             if empty and have_good_batch:
#                 num_rejected += 1
#                 logger.debug(
#                     "reject empty mask"
#                     # "reject empty gt at %s",
#                     # batch.arrays[self.gt].spec.roi,
#                 )
#                 if timing.elapsed() > report_next_timeout:
#                     logger.warning(
#                         "rejected %d batches, been waiting for a good one " "since %ds",
#                         num_rejected,
#                         report_next_timeout,
#                     )
#                     report_next_timeout *= 2
#                 continue
#             else:
#                 break
#
#         timing.stop()
#         batch.profiling_stats.add(timing)
#
#         return batch


import logging
import random
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.profiling import Timing
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)


class MaskReject(BatchFilter):
    """
    Reject batches based on the masked-in vs. masked-out ratio.

    Args:
        mask (Key): The key for the mask array.
        raw (Key): The key for the raw array.
        p (float, optional): The probability of accepting a batch with an empty mask. Defaults to 0.5.
        background (int, optional): Value representing the background in the mask. Defaults to 0.
    """

    def __init__(self, mask, raw, p=0.5, background=0):
        self.mask = mask
        self.raw = raw
        self.p = p
        self.background = background

    def setup(self):
        assert len(self.get_upstream_providers()) == 1, "Only one upstream provider is supported."
        self.upstream_provider = self.get_upstream_providers()[0]

    def provide(self, request):
        random.seed(request.random_seed)

        num_rejected = 0
        report_next_timeout = 10

        timing = Timing(self)
        timing.start()

        # have_good_batch = random.random() < self.p

        while True:
            batch = self.upstream_provider.request_batch(request)

            # Get raw and mask ROIs
            raw_roi = batch.arrays[self.raw].roi
            mask_roi = batch.arrays[self.mask].roi

            # Calculate the corresponding mask ROI for the raw ROI
            mask_scale = Coordinate(
                raw_roi.voxel_size[d] // mask_roi.voxel_size[d] for d in range(len(raw_roi.voxel_size))
            )
            mask_start = Coordinate(
                (raw_roi.offset[d] - mask_roi.offset[d]) // mask_roi.voxel_size[d]
                for d in range(len(raw_roi.offset))
            )
            mask_shape = Coordinate(
                raw_roi.shape[d] // mask_roi.voxel_size[d] for d in range(len(raw_roi.shape))
            )
            relevant_mask_roi = mask_roi.with_offset(mask_start).with_shape(mask_shape)

            # Get the relevant mask data
            mask_data = batch.arrays[self.mask].data
            relevant_mask = mask_data[
                tuple(slice(mask_start[d], mask_start[d] + mask_shape[d]) for d in range(len(mask_start)))
            ]

            # Check if the relevant mask is empty
            empty = relevant_mask.sum() == 0

            if empty:
                num_rejected += 1
                logger.debug("Rejected batch with empty mask.")
                if timing.elapsed() > report_next_timeout:
                    logger.warning(
                        "Rejected %d batches, been waiting for a good one for %ds",
                        num_rejected,
                        report_next_timeout,
                    )
                    report_next_timeout *= 2
                continue
            else:
                break

        timing.stop()
        batch.profiling_stats.add(timing)
        return batch
