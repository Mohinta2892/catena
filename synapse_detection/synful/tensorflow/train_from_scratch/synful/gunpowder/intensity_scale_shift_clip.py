import numpy as np

from gunpowder import BatchFilter


class IntensityScaleShiftClip(BatchFilter):
    '''Scales the intensities of a batch by ``scale``, then adds ``shift``.
    Optionally, also clips arrays.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        scale (``float``):
        shift (``float``):

            The shift and scale to apply to ``array``.
        clip (``tuple``):

            Clip_min and clip_max value, clipping applied after scaling.

    '''

    def __init__(self, array, scale, shift, clip=None):
        self.array = array
        self.scale = scale
        self.shift = shift
        self.clip = clip

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        raw = batch.arrays[self.array]
        raw.data = raw.data * self.scale + self.shift
        # raw.data = np.round(raw.data)
        if self.clip is not None:
            raw.data = np.clip(raw.data, self.clip[0], self.clip[1])
