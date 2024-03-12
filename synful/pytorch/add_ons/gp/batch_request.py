import copy
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.graph import GraphKey
from .points import PointsKey
from .points_spec import PointsSpec
from gunpowder.graph_spec import GraphSpec

from warnings import warn
import time


class BatchRequest(ProviderSpec):
    """A collection of (possibly partial) :class:`ArraySpec` and
    :class:`GraphSpec` forming a request.

    Inherits from :class:`ProviderSpec`.

    Additional Kwargs:

        random_seed (``int``):

            The random seed that will be associated with this batch to
            guarantee deterministic and repeatable batch requests.

    """

    def __init__(self, *args, random_seed=None, **kwargs):
        self._random_seed = random_seed
        super().__init__(*args, **kwargs)

    def add(self, key, shape, voxel_size=None, directed=None, placeholder=False):
        """Convenience method to add an array or graph spec by providing only
        the shape of a ROI (in world units).

        A ROI with zero-offset will be generated. If more than one request is
        added, the ROIs with smaller shapes will be shifted to be centered in
        the largest one.

        Args:

            key (:class:`ArrayKey` or :class:`GraphKey`):

                The key for which to add a spec.

            shape (:class:`Coordinate`):

                A tuple containing the shape of the desired roi

            voxel_size (:class:`Coordinate`):

                A tuple contening the voxel sizes for each corresponding
                dimension
        """

        if isinstance(key, ArrayKey):
            spec = ArraySpec(placeholder=placeholder)
        elif isinstance(key, GraphKey):
            spec = GraphSpec(placeholder=placeholder, directed=directed)
        elif isinstance(key, PointsKey):
            """ added for Synful"""
            spec = PointsSpec()
        else:
            raise RuntimeError("Only ArrayKey or GraphKey or PointsKey [optional for Synful] can be added.")

        spec.roi = Roi((0,) * len(shape), shape)

        if voxel_size is not None:
            spec.voxel_size = voxel_size

        self[key] = spec
        self.__center_rois()

    def copy(self):
        """Create a copy of this request."""
        request_copy = copy.deepcopy(self)
        request_copy._update_random_seed()
        return request_copy

    def is_deterministic(self):
        """Return true if a random seed has been set for this request."""
        return self._random_seed is not None

    @property
    def random_seed(self):
        if not self.is_deterministic():
            return int(time.time() * 1e6) % (2 ** 32)
        else:
            return self._random_seed

    def _update_random_seed(self):
        if not self.is_deterministic():
            return
        self._random_seed = hash((self._random_seed + 1) ** 2) % (2 ** 32)

    def __center_rois(self):
        """Ensure that all ROIs are centered around the same location."""

        total_roi = self.get_total_roi()
        if total_roi is None:
            return

        center = total_roi.center

        for specs_type in [self.array_specs, self.graph_specs]:
            for key in specs_type:
                roi = specs_type[key].roi
                specs_type[key].roi = roi.shift(center - roi.center)

    def update_with(self, request):
        """Update current request with another"""

        assert isinstance(request, BatchRequest)

        merged = self.copy()

        for key, spec in request.items():
            if key not in merged:
                merged[key] = spec
            else:
                merged[key].update_with(spec)

        return merged

    def merge(self, request):
        """Merge another request with current request"""
        warn(
            "merge is deprecated! please use update_with "
            "as it accounts for spec metadata"
        )
        assert isinstance(request, BatchRequest)

        merged = self.copy()

        for key, spec in request.items():
            if key not in merged:
                merged[key] = spec
            else:
                if isinstance(spec, ArraySpec) and merged[key].nonspatial:
                    merged[key] = spec
                else:
                    merged[key].roi = merged[key].roi.union(spec.roi)

        return merged

    def __eq__(self, other):
        """
        Override equality check to allow batche requests with different
        seeds to still be checked. Otherwise equality check should
        never succeed.
        """

        if isinstance(other, self.__class__):
            other_dict = copy.deepcopy(other.__dict__)
            self_dict = copy.deepcopy(self.__dict__)
            other_dict.pop("_random_seed")
            self_dict.pop("_random_seed")
            return self_dict == other_dict
        return NotImplemented
