import logging

import numpy as np

from gunpowder import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.morphology import enlarge_binary_map
# from .points_spec import PointsSpec
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
from scipy.spatial import KDTree

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AddPartnerVectorMap(BatchFilter):
    '''Create an array with offset vectors pointing to target points.



    Args:

        src_points (:class:``PointsKeys``):
            Source points around which vectors are created.

        trg_points (:class:``PointsKeys``):
            Target points to which created vectors point to.

        array (:class:``ArrayKey``):
            The key of the array to create.

        radius (``tuple`` of ``float``):
            Radius of the ball around source points in which vectors are
            created. If two or more src nodes are in distance < radius,
            voxels of overlapping regions are assigned the closest src point to.

        trg_context (``tuple`` of ``int`` or ``int``):
            n-dim tuple which defines padding of trg_points request in world
            units to create src vectors that point to target locations
            outside src roi.

        mask (:class:`ArrayKey`, optional):
            Used to mask the rasterization of source points. The array is assumed to
            contain discrete labels. The object id at the specific point being
            rasterized is used to intersect the rasterization to keep it inside
            the specific object.

        pointmask (:class:`ArrayKey`, optional):
            Key for a pointmask. If provided, an array is created,
            in which all blob regions for which vectors have been calculated,
            is marked with a 1. Array has same spec as array.


        array_spec (:class:``ArraySpec``, optional):
            The spec of the array to create. Use this to set the datatype and
            voxel size.
    '''

    def __init__(self, src_points, trg_points, array, radius, trg_context,
                 mask=None, pointmask=None, array_spec=None):

        self.src_points = src_points
        self.trg_points = trg_points
        self.array = array
        self.radius = np.array([radius]).flatten().astype(np.float32)
        self.trg_context = np.array([trg_context]).flatten().astype(int)
        self.mask = mask
        self.pointmask = pointmask
        if array_spec is None:
            self.array_spec = ArraySpec()
        else:
            self.array_spec = array_spec

    def setup(self):

        src_roi = self.spec[self.src_points].roi

        if self.array_spec.voxel_size is None:
            # self.array_spec.voxel_size = Coordinate((1,) * src_roi.dims())
            self.array_spec.voxel_size = Coordinate((1,) * src_roi.dims)

        if self.array_spec.dtype is None:
            self.array_spec.dtype = np.float32

        self.array_spec.roi = src_roi.copy()
        self.provides(
            self.array,
            self.array_spec)
        if self.pointmask is not None:
            self.provides(
                self.pointmask,
                self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):

        # For src point, use radius to determine the context.
        context = np.ceil(self.radius).astype(int)

        # dims = self.array_spec.roi.dims()
        # this is no longer a method, but an attribute/property
        dims = self.array_spec.roi.dims
        if len(context) == 1:
            context = context.repeat(dims)

        # request points in a larger area
        src_roi = request[self.array].roi.grow(
            Coordinate(context),
            Coordinate(context))

        # however, restrict the request to the points actually provided
        src_roi_prov = src_roi.intersect(self.spec[self.src_points].roi)
        request[self.src_points] = GraphSpec(roi=src_roi_prov)

        # For trg points, use custom option.
        context = self.trg_context
        if len(context) == 1:
            context = context.repeat(dims)

        # request points in a larger area
        trg_roi = src_roi.grow(
            Coordinate(context),
            Coordinate(context))

        # however, restrict the request to the points actually provided
        trg_roi = trg_roi.intersect(self.spec[self.trg_points].roi)
        request[self.trg_points] = GraphSpec(roi=trg_roi)

        if self.mask is not None:
            mask_voxel_size = self.spec[self.mask].voxel_size
            assert self.spec[self.array].voxel_size == mask_voxel_size, (
                "Voxel size of mask and rasterized volume need to be equal")

            # Commenting this out does not code to crash.
            # Todo: See the repercusions in model training
            # new_mask_roi = src_roi.snap_to_grid(mask_voxel_size)
            # # Restrict request to array provided.
            # new_mask_roi = new_mask_roi.intersect(self.spec[self.mask].roi)
            # if self.mask in request:
            #     request[self.mask].roi = \
            #         request[self.mask].roi.union(new_mask_roi)
            # else:
            #     request[self.mask] = \
            #         ArraySpec(roi=new_mask_roi)

            mask_roi = src_roi.snap_to_grid(mask_voxel_size)
            request[self.mask] = ArraySpec(roi=mask_roi)

    def process(self, batch, request):

        # src_points = batch.points[self.src_points]
        src_points = batch.graphs[self.src_points]  # points now graphs
        voxel_size = self.spec[self.array].voxel_size

        # get roi used for creating the new array (points_roi does not
        # necessarily align with voxel size)
        enlarged_vol_roi = src_points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_vol_roi.get_begin() / voxel_size
        shape = enlarged_vol_roi.get_shape() / voxel_size
        data_roi = Roi(offset, shape)

        logger.debug("Src points in %s", src_points.spec.roi)
        # for i, point in src_points.data.items():
        for i, point in enumerate(src_points.nodes):
            logger.debug("%d, %s", i, point.location)
        logger.debug("Data roi in voxels: %s", data_roi)
        logger.debug("Data roi in world units: %s", data_roi * voxel_size)

        mask_array = None if self.mask is None else batch.arrays[
            self.mask].crop(enlarged_vol_roi)

        partner_vectors_data, pointmask = self.__draw_partner_vectors(
            src_points,
            # batch.points[self.trg_points],
            batch.graphs[self.trg_points],
            data_roi,
            voxel_size,
            enlarged_vol_roi.get_begin(),
            self.radius,
            mask_array)

        # if np.sum(partner_vectors_data) == 0.:
        #
        #     print("----------------------------------")
        #     print(f"sum of partner_vectors_data is zero")
        #     # # instead of returning setting self.point_mask = None
        #     print(f"Sum of pointmask is {np.sum(pointmask)}")
        #     print("----------------------------------")
        #     # quit()

        # print(request)

        # create array and crop it to requested roi
        spec = self.spec[self.array].copy()
        spec.roi = data_roi * voxel_size
        partner_vectors = Array(
            data=partner_vectors_data,
            spec=spec)
        logger.debug("Cropping partner vectors to %s", request[self.array].roi)
        batch.arrays[self.array] = partner_vectors.crop(request[self.array].roi)

        if self.pointmask is not None and self.pointmask in request:
            spec = self.spec[self.array].copy()
            spec.roi = data_roi * voxel_size
            pointmask = Array(
                data=np.array(pointmask, dtype=spec.dtype),
                spec=spec)
            batch.arrays[self.pointmask] = pointmask.crop(
                request[self.pointmask].roi)

        # restore requested ROI of src and target points.
        if self.src_points in request:
            self.__restore_points_roi(request, self.src_points,
                                      # batch.points[self.src_points
                                      batch.graphs[self.src_points
                                      ])
        if self.trg_points in request:
            self.__restore_points_roi(request, self.trg_points,
                                      # batch.points[self.trg_points]
                                      batch.graphs[self.trg_points]
                                      )
        # restore requested objectmask
        if self.mask is not None:
            # this one is cropping the gt_neurons back to the original request size
            # then this gets passed to rasterize_settings and graphs where the expectation
            batch.arrays[self.mask] = batch.arrays[self.mask].crop(
                request[self.mask].roi)

    def __restore_points_roi(self, request, points_key, points):
        request_roi = request[points_key].roi
        points.spec.roi = request_roi
        # For Points and PointsSpec with Gunpowder 0.3.0 dev version
        # points.data = {i: p for i, p in points.data.items() if
        #                request_roi.contains(p.location)}

        # For Graphs and GraphsSpec with > Gunpowder 1.3.0 dev version
        points.data = {i: p for i, p in enumerate(points.nodes) if
                       request_roi.contains(p.location)}

        # for i, p in points.data.items():
        #     if not request_roi.contains(p.location):
        #         del points.data[i]

    def __draw_partner_vectors(self, src_points, trg_points, data_roi,
                               voxel_size, offset, radius, mask=None):

        # 3D: z, y, x
        shape = data_roi.get_shape()
        logger.debug('data roi %s' % data_roi)
        d, h, w = shape

        # 4D: c, z, y, x (c=[0, 1, 2])
        coords = np.array(
            # 3D: z, y, x
            np.meshgrid(
                np.arange(0, d),
                np.arange(0, h),
                np.arange(0, w),
                indexing='ij'),
            dtype=np.float32)

        # 4D: c, z, y, x
        coords[0, :] *= voxel_size[0]
        coords[1, :] *= voxel_size[1]
        coords[2, :] *= voxel_size[2]
        coords[0, :] += offset[0]
        coords[1, :] += offset[1]
        coords[2, :] += offset[2]

        target_vectors = np.zeros_like(coords)

        logger.debug(
            "Adding vectors for %d points...",
            # len(src_points.data)
            src_points.num_vertices()
        )

        # For each src point, get a point mask.
        union_mask = np.zeros(shape, dtype=np.int32)
        point_masks = []
        points_p = []
        targets = []

        # for point_id, point in src_points.data.items():
        # For Graphs and GraphsSpec with > Gunpowder 1.3.0 dev version
        for point_id, point in enumerate(src_points.nodes):
            # get the voxel coordinate, 'Coordinate' ensures integer
            v = Coordinate(point.location / voxel_size)

            if not data_roi.contains(v):
                logger.debug(
                    "Skipping point at %s outside of requested data ROI",
                    v)
                continue

            assert len(
                point.partner_ids) == 1, \
                'AddPartnerVectorMap only implemented for single target point ' \
                'per src point'
            trg_id = point.partner_ids[0]

            # if trg_id not in trg_points.data:
            if not trg_points.contains(node_id=trg_id):
                logger.warning(
                    "target %d of %d not in trg points",
                    trg_id,
                    point_id)
                continue

            # target = trg_points.data[trg_id]
            target = trg_points.node(id=trg_id)
            if not trg_points.spec.roi.contains(target.location):
                logger.warning(
                    "target %d of %d not in target roi: %s",
                    trg_id,
                    point_id, trg_points.spec.roi)
                continue

            # get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            if mask is not None:
                label = mask.data[v]
                object_mask = mask.data == label
            logger.debug(
                "Rasterizing point %s at %s",
                point.location,
                point.location / voxel_size - data_roi.get_begin())

            # numpy.bool is numpy.bool_ from <  NumPy 1.20; we use builtin bool instead
            point_mask = np.zeros(shape, dtype=bool)
            point_mask[v] = 1

            # print(f"voxel coordinate relative to output array start {v} and point_mask shape={shape}")
            # print("Sum of point_mask after assignment", np.sum(point_mask))

            enlarge_binary_map(
                point_mask,
                radius,
                voxel_size,
                in_place=True)

            if mask is not None:
                point_mask &= object_mask
            union_mask += np.array(point_mask, dtype=np.int32)
            point_masks.append(point_mask)
            targets.append(target)  # change
            points_p.append(v * voxel_size)

            # plt.close("all")
            # plt.imshow(point_mask[0, ...], cmap='gray')
            # plt.show()

        assert len(targets) == len(points_p) == len(point_masks)
        if len(points_p) == 0:
            return target_vectors, np.array(union_mask, dtype=bool)  # Leave early if there are no points.
        for point_mask in point_masks:
            point_mask[union_mask > 1] = False  # Remove overlap regions.
            # print("Sum of point_mask after removing overlaps", np.sum(union_mask))
        intersect_points = np.where(union_mask > 1)
        logger.debug('#voxels of overlapping src blobs:{}'.format(
            len(intersect_points[0])))

        # Assign overlapping voxels to their closest src node.
        kd = KDTree(points_p)
        for intersect_point in zip(*intersect_points):
            p = Coordinate(intersect_point) * voxel_size
            dist, node_id = kd.query(p)
            point_masks[node_id][p / voxel_size] = True

        # Calculate actual vectors with src blobs corrected for overlaps.
        for ii, point_mask in enumerate(point_masks):
            target = targets[ii]
            target_vectors[0][point_mask] = target.location[0] - coords[0][
                point_mask]
            target_vectors[1][point_mask] = target.location[1] - coords[1][
                point_mask]
            target_vectors[2][point_mask] = target.location[2] - coords[2][
                point_mask]

        # numpy.bool deprecated from  NumPy 1.20; favoured builtin bool
        # if np.sum(np.array(union_mask, dtype=np.bool)) == 0:
        # if np.sum(np.array(union_mask, dtype=bool)) == 0:
        #     print('union mask is zero', np.sum(np.array(union_mask, dtype=bool)))
        #     print('target vectors', target_vectors)
        # print(f'Max target vector value: {np.max(target_vectors)}')
        # print(f'shape of point mask: {np.array(union_mask, dtype=bool).shape}')
        return target_vectors, np.broadcast_to(np.array(union_mask, dtype=bool), (3,)+np.array(union_mask, dtype=bool).shape)
