import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.nodes.batch_provider import BatchProvider

from cloudvolume import CloudVolume
from scipy import ndimage

logger = logging.getLogger(__name__)


class CloudVolumeSource(BatchProvider):
    '''A source for CloudVolume.

    Provides array from CloudVolume.

    Args:

        cloudvolume_url (``string``):

            The cloudvolume url from which to load the array.

        array_key (:class:`ArrayKey`):

            Array key for the array.

        array_spec (:class:`ArraySpec`, optional):

            Array spec to overwrite
            the array spec automatically determined from the cloudvolume meta information. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''

    def __init__(
            self,
            cloudvolume_url,
            array_key,
            mip=0,
            array_spec=None):

        self.cloudvolume_url = cloudvolume_url
        self.mip = mip
        self.array_key = array_key
        self.array_spec = array_spec
        self.ndims = None

    def setup(self):
        cv = CloudVolume(self.cloudvolume_url, use_https=True, mip=self.mip)
        spec = self.__read_spec(cv)
        logger.debug(f'Spec is {spec}')

        self.array_spec = spec
        self.provides(self.array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        cv = CloudVolume(self.cloudvolume_url, use_https=True, mip=self.mip)

        request_spec = request.array_specs[self.array_key]
        array_key = self.array_key
        logger.debug("Reading %s in %s...", array_key, request_spec.roi)

        voxel_size = self.array_spec.voxel_size

        # scale request roi to voxel units
        dataset_roi = request_spec.roi / voxel_size

        # shift request roi into dataset
        dataset_roi = dataset_roi - self.spec[
            array_key].roi.get_offset() / voxel_size

        # create array spec
        array_spec = self.array_spec.copy()
        array_spec.roi = request_spec.roi
        # array_spec.voxel_size = array_spec.voxel_size

        # add array to batch
        batch.arrays[array_key] = Array(
            self.__read(cv, dataset_roi),
            array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_voxel_size(self, cv=None):
        try:
            if cv is None:
                cv = CloudVolume(self.cloudvolume_url, use_https=True,
                                 mip=self.mip)
            return Coordinate(cv.resolution[::-1])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def _get_offset(self, cv):
        try:
            return Coordinate(cv.bounds.minpt[::-1]) * self._get_voxel_size(cv)
        except Exception:  # todo: make specific when z5py supports it
            return None

    def __read_spec(self, cv):

        # dataset = data_file[ds_name]

        dims = Coordinate(cv.bounds.maxpt[::-1]) * self._get_voxel_size(cv)

        if self.ndims is None:
            self.ndims = cv.bounds.ndim
        else:
            assert self.ndims == len(dims)

        if self.array_spec is not None:
            spec = self.array_spec.copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = self._get_voxel_size(cv)
            if voxel_size is None:
                voxel_size = Coordinate((1,) * self.ndims)
                logger.warning(
                    "WARNING: File %s does not contain resolution information "
                    "for %s , voxel size has been set to %s. This "
                    "might not be what you want.",
                    self.filename, array_key, spec.voxel_size)
            spec.voxel_size = voxel_size

        if spec.roi is None:
            offset = self._get_offset(cv)
            if offset is None:
                offset = Coordinate((0,) * self.ndims)

            spec.roi = Roi(offset, dims * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == cv.dtype, (
                        "dtype %s provided in array_specs for %s, "
                        "but differs from cloudvolume dtype %s" %
                        (self.array_spec.dtype,
                         self.array_key, dataset.dtype))
        else:
            spec.dtype = cv.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           ". Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           self.array_key, spec.dtype,
                           spec.interpolatable)

        return spec

    def __read(self, cv, roi):
        arr = np.squeeze(np.asarray(cv[roi.get_bounding_box()[::-1]]), axis=-1)
        arr = arr.transpose()
        return arr

    def __repr__(self):

        return self.cloudvolume_url
