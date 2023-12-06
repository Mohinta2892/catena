"""
A few extra gunpowder based filters:
EnsureUInt8 : Casts ndarray to `np.uint8` (intensity range 0-255)
InvertAffPred : Inverts the affinity matrices. Was implemented because plant-seg scripts work with affinities
denoted by white boundaries and black intracellular space.
MergeAffinityChannels: Sums (Change code to mean if needed) the affinity channels into one.
ChooseMaxAffinityValue: Chooses max affinities over all available channels

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University, UK
"""
import os
import logging

import numpy as np
from gunpowder import *
import skimage

# get the same logger that is defined in parent script
logger = logging.getLogger(__name__)


class RandomLocationNonEmpty(RandomLocation):
    """Override accepts to reject batches that contain blank labels
    """

    def __init__(self, **kwargs):
        super(RandomLocationNonEmpty, self).__init__(**kwargs)

    def accepts(self, request):
        batch = super().get_upstream_provider().request_batch(request)
        for key, spec in request.items():
            non_empty = np.sum(batch[key].data) > 0.
            if not non_empty:
                return False

        return True


class EnsureUInt8(BatchFilter):
    """
    Copied from existing Funkelab's lsd_experiments repository, where it has been used in certain setups.
    Source: https://github.com/funkelab/lsd_experiments
    """

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        batch[self.array].data = (batch[self.array].data * 255.0).astype(np.uint8)


class InvertAffPred(BatchFilter):
    """
    Inverts the affinities array using `skimage`.
    Design Choice: We need this if we are to run watershed and agglomeration via the plantseg pipeline.
    """

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        batch[self.array].data = skimage.util.invert(batch[self.array].data)


class MergeAffinityChannels(BatchFilter):
    """This is wrong, you will see a lot of artefacts wherein the mitochondrial membranes also get captured if you just
    sum all pixel values across channels. To suppress this you could mean them.
    This code is kept as note for design choice, even though it would be obvious to some people of why you should not
    implement as such"""

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        # data shape generally:: C x D(Z) x H(Y) x W(X)
        batch[self.array].data = np.nansum(batch[self.array].data, axis=0)


class ChooseMaxAffinityValue(BatchFilter):
    """Choose the maximum value of affinity per pixel across all channels"""

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        # data shape generally::  C x D(Z) x H(Y) x W(X)
        batch[self.array].data = np.nanmax(batch[self.array].data, axis=0)


class ExampleSourceCrop(BatchProvider):
    def setup(self):
        self.provides(
            ArrayKeys.RAW,
            ArraySpec(roi=Roi((200, 20, 20), (1800, 180, 180)), voxel_size=(20, 2, 2)),
        )

        self.provides(
            GraphKeys.PRESYN, GraphSpec(roi=Roi((200, 20, 20), (1800, 180, 180)))
        )

    def provide(self, request):
        pass


class SnapRawToLabelsGrid(BatchFilter):
    """Choose the maximum value of affinity per pixel across all channels"""

    def __init__(self, input_raw, output_raw, labels, voxel_size):
        self.input_raw = input_raw
        self.output_raw = output_raw
        self.labels = labels
        self.voxel_size = voxel_size

    def prepare(self, request):
        pass

    def process(self, batch, request):
        outputs = Batch()
        # calculate label start and end
        start = request[self.labels].roi.get_begin() / self.voxel_size
        end = request[self.labels].roi.get_end() / self.voxel_size

        snap_roi = request[self.output_raw].roi
        spec = self.spec[self.output_raw].copy()
        spec.roi = snap_roi
        # data shape generally::  Batch x Channels x D(Z) x H(Y) x W(X)
        if len(batch[self.input_raw].data.shape) == 4:  # 2D
            outputs.arrays[self.output_raw] = Array(batch[self.input_raw].data[:, :, start[0]:end[0], start[1]: end[1]],
                                                    spec)
        elif len(batch[self.array].data.shape) == 5:  # 3D
            outputs.arrays[self.output_raw] = Array(batch[self.input_raw].data[:, :, start[0]:end[0], start[1]: end[1],
                                                    start[2]: end[2]], spec)
        else:
            raise RuntimeError("Wrong shape of input")

        return outputs


class EnsureNonEmptyLabel(BatchFilter):
    """RandomLocation() can fetch completely empty labels,
     we wish to discard these examples completely from a batch.
     This is slow, since it involves a for to go over all slices in a batch.
     """

    def __init__(self, raw, labels, mask=None, background_value=0):
        self.raw = raw
        self.labels = labels
        self.mask = mask
        self.background_value = background_value

    def setup(self):
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):
        # get the batch here
        batch = self.upstream_provider.request_batch(request)
        # data shape generally:: B x D(Z) x H(Y) x W(X)
        # empty_pos = []
        print(batch)
        # for i in range(batch[self.labels].data.shape):
        #     if np.sum(batch[self.labels].data[i, ...]) == self.background_value:
        #         empty_pos.append(i)
        # if len(empty_pos) > :
        #     # batch[self.raw].data = np.delete(batch[self.raw].data, empty_pos, axis=0)
        #     # batch[self.labels].data = np.delete(batch[self.labels].data, empty_pos, axis=0)
        #     # if self.mask is not None:
        #     #     batch[self.mask].data = np.delete(batch[self.mask].data, empty_pos, axis=0)

        return batch
