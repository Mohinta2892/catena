import logging
import unittest

import numpy as np

from gunpowder import *
from gunpowder.contrib.points import PreSynPoint, PostSynPoint
from gunpowder.points import Points
from synful.gunpowder import AddPartnerVectorMap

logging.basicConfig(level=logging.INFO)
logging.getLogger('synful.gunpowder').setLevel(logging.INFO)


class PointTestSource3D(BatchProvider):
    def __init__(self, points, partners, objectmask=None, voxel_size=None):
        self.voxel_size = voxel_size

        self.points = points

        self.partners = partners
        self.objectmask = objectmask

    def setup(self):

        self.provides(
            PointsKeys.PRESYN,
            PointsSpec(roi=Roi((0, 0, 0), (200, 200, 200))))

        self.provides(
            PointsKeys.POSTSYN,
            PointsSpec(roi=Roi((0, 0, 0), (200, 200, 200))))
        if self.objectmask is not None:
            self.provides(
                ArrayKeys.OBJECTMASK,
                ArraySpec(
                    roi=Roi((0, 0, 0), (200, 200, 200)),
                    voxel_size=self.voxel_size))

    def provide(self, request):

        batch = Batch()

        roi_points = request[PointsKeys.PRESYN].roi
        trg_points = request[PointsKeys.POSTSYN].roi

        # get all pre points inside the requested ROI
        pre_points = {}
        post_points = {}
        syn_id = 0
        for pre_id, post_id in self.partners:
            loc = self.points[pre_id]
            if roi_points.contains(loc):
                pre_point = PreSynPoint(location=loc,
                                        partner_ids=[post_id],
                                        location_id=pre_id, synapse_id=syn_id)
                pre_points[pre_id] = pre_point
            loc = self.points[post_id]
            if trg_points.contains(loc):
                post_point = PostSynPoint(location=loc,
                                          partner_ids=[pre_id],
                                          location_id=post_id,
                                          synapse_id=syn_id)

                post_points[post_id] = post_point
            syn_id += 1

        batch.points[PointsKeys.PRESYN] = Points(
            pre_points,
            PointsSpec(roi=roi_points))
        batch.points[PointsKeys.POSTSYN] = Points(
            post_points,
            PointsSpec(roi=trg_points))

        if ArrayKeys.OBJECTMASK in request:
            roi_array = request[ArrayKeys.OBJECTMASK].roi

            spec = self.spec[ArrayKeys.OBJECTMASK].copy()

            spec.roi = roi_array
            batch.arrays[ArrayKeys.OBJECTMASK] = Array(
                self.objectmask[(roi_array / self.voxel_size).to_slices()],
                spec=ArraySpec(roi=roi_array, voxel_size=self.voxel_size))

        return batch


class TestAddPartnerVectorMap(unittest.TestCase):
    def test_output_basics(self):
        vectormap = ArrayKey('VECTOR_MAP')
        presyn = PointsKey('PRESYN')
        postsyn = PointsKey('POSTSYN')

        voxel_size = Coordinate((10, 10, 10))
        spec = ArraySpec(voxel_size=voxel_size)

        add_vector_map = AddPartnerVectorMap(
            src_points=presyn,
            trg_points=postsyn,
            array=vectormap,
            radius=1,  # 10 voxels,
            trg_context=20,  # 10 voxels,,
            array_spec=spec
        )

        points = {
            1: Coordinate((40, 40, 40)),
            2: Coordinate((90, 90, 40)),
            3: Coordinate((70, 70, 70)),
        # should be ignored, as partner is far outside
            4: Coordinate((300, 300, 300)),
        }

        pipeline = (
                PointTestSource3D(points=points, partners=[(1, 2), (3, 4)],
                                  voxel_size=voxel_size) +
                add_vector_map
        )

        request = BatchRequest()

        roi = Roi((40, 40, 40), (80, 80, 80))

        request[presyn] = PointsSpec(roi=roi)
        request[postsyn] = PointsSpec(roi=roi)
        request[vectormap] = ArraySpec(roi=roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        res = batch[vectormap]
        self.assertTrue((res.data[:, 0, 0, 0] == np.array([50, 50, 0])).all())
        # The rest should be all zero.
        self.assertEqual(np.count_nonzero(res.data[:, 1:, 1:, 1:]), 0)

    def test_output_outside_roi(self):
        vectormap = ArrayKey('VECTOR_MAP')
        presyn = PointsKey('PRESYN')
        postsyn = PointsKey('POSTSYN')

        voxel_size = Coordinate((5, 5, 5))
        spec = ArraySpec(voxel_size=voxel_size)

        add_vector_map = AddPartnerVectorMap(
            src_points=presyn,
            trg_points=postsyn,
            array=vectormap,
            radius=30,  # 10 voxels,
            trg_context=20,  # 10 voxels,,
            array_spec=spec
        )

        points = {
            1: Coordinate((30, 30, 30)),
        # should be included, since radius is large enough to reach into ROI.
            2: Coordinate((90, 90, 40)),
            3: Coordinate((70, 70, 70)),
        # should be included, since partner is within context
            4: Coordinate((90, 90, 50)),
        }

        pipeline = (
                PointTestSource3D(points=points, partners=[(1, 2), (3, 4)]) +
                add_vector_map
        )

        request = BatchRequest()

        roi = Roi((40, 40, 40), (80, 80, 80))

        request[presyn] = PointsSpec(roi=roi)
        request[postsyn] = PointsSpec(roi=roi)
        request[vectormap] = ArraySpec(roi=roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        res = batch[vectormap]
        self.assertTrue((res.data[:, 0, 0, 0] == np.array([50, 50, 0])).all())
        self.assertTrue((res.data[:, 6, 6, 6] == np.array([20, 20, -20])).all())

    def test_intersecting_src_blobs(self):
        vectormap = ArrayKey('VECTOR_MAP')
        objectmask = ArrayKey('OBJECTMASK')
        presyn = PointsKey('PRESYN')
        postsyn = PointsKey('POSTSYN')
        pointmask = ArrayKey('POINTMASK')

        voxel_size = Coordinate((5, 5, 5))
        spec = ArraySpec(voxel_size=voxel_size)

        add_vector_map = AddPartnerVectorMap(
            src_points=presyn,
            trg_points=postsyn,
            array=vectormap,
            radius=60,  # 10 voxels,
            trg_context=20,  # 10 voxels,,
            array_spec=spec
        )

        points = {
            1: Coordinate((20, 20, 50)),  # point1 in radius distance to point3
            2: Coordinate((0, 0, 0)),
            3: Coordinate((60, 60, 50)),
            4: Coordinate((50, 50, 50)),
        }

        pipeline = (
                PointTestSource3D(points=points, partners=[(1, 2), (3, 4)]) +
                add_vector_map
        )

        request = BatchRequest()

        roi = Roi((0, 0, 0), (200, 200, 200))

        request[presyn] = PointsSpec(roi=roi)
        request[postsyn] = PointsSpec(roi=roi)
        request[vectormap] = ArraySpec(roi=roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        res = batch[vectormap]
        test1 = Coordinate((30, 30, 50)) / voxel_size
        test3 = Coordinate((50, 50, 50)) / voxel_size
        # testpoint1 should be close to point1
        self.assertTrue((res.data[:, test1[0], test1[1], test1[2]] == np.array(
            [-30, -30, -50])).all())
        # testpoint3 should be close to point3
        self.assertTrue((res.data[:, test3[0], test3[1], test3[2]] == np.array(
            [0, 0, 0])).all())

        object_mask_ar = np.ones(
            np.array((200 / 5, 200 / 5, 200 / 5), dtype=np.int))

        object_mask_ar[points[3] / voxel_size] = 2

        add_vector_map = AddPartnerVectorMap(
            src_points=presyn,
            trg_points=postsyn,
            array=vectormap,
            radius=50,  # 10 voxels,
            trg_context=20,  # 10 voxels,,
            array_spec=spec,
            mask=objectmask,
            pointmask=pointmask
        )

        pipeline = (
                PointTestSource3D(points=points,
                                  partners=[(1, 2), (3, 4)],
                                  objectmask=object_mask_ar,
                                  voxel_size=voxel_size) +
                add_vector_map
        )

        request[presyn] = PointsSpec(roi=roi)
        request[postsyn] = PointsSpec(roi=roi)
        request[vectormap] = ArraySpec(roi=roi)
        request[objectmask] = ArraySpec(roi=roi)
        request[pointmask] = ArraySpec(roi=roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)
        res = batch[vectormap]
        # 30, 30, 50 should be close to point1
        self.assertTrue((res.data[:, test1[0], test1[1], test1[2]] == np.array(
            [-30, -30, -50])).all())
        # 50, 50, 50 should also be close to point1 because of object mask
        self.assertTrue((res.data[:, test3[0], test3[1], test3[2]] == np.array(
            [-50, -50, -50])).all())

        # Also test pointmask
        self.assertTrue(batch[pointmask].data[test1[0], test1[1], test1[2]] == 1)


if __name__ == '__main__':
    unittest.main()
