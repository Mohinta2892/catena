import logging
import os
import shutil
import tempfile
import unittest

import gunpowder as gp
import numpy as np
import synful
from synful.gunpowder import ExtractSynapses

logging.basicConfig(level=logging.INFO)
logging.getLogger('synful.gunpowder').setLevel(logging.INFO)

parameter_dic = {
    "extract_type": "cc",
    "cc_threshold": 0.50,
    "loc_type": "edt",
    "score_thr": 0,
    "score_type": "sum",
    "nms_radius": None
}

parameters = synful.detection.SynapseExtractionParameters(
    extract_type=parameter_dic['extract_type'],
    cc_threshold=parameter_dic['cc_threshold'],
    loc_type=parameter_dic['loc_type'],
    score_thr=parameter_dic['score_thr'],
    score_type=parameter_dic['score_type'],
    nms_radius=parameter_dic['nms_radius']
)

gp.ArrayKey('M_PRED')
gp.ArrayKey('D_PRED')
gp.PointsKey('PRESYN')
gp.PointsKey('POSTSYN')


class TestSource(gp.BatchProvider):

    def __init__(self, m_pred, d_pred, voxel_size):
        self.voxel_size = voxel_size
        self.m_pred = m_pred
        self.d_pred = d_pred

    def setup(self):
        self.provides(
            gp.ArrayKeys.M_PRED, gp.ArraySpec(
                roi=gp.Roi((0, 0, 0), (200, 200, 200)),
                voxel_size=self.voxel_size,
                interpolatable=False))
        self.provides(
            gp.ArrayKeys.D_PRED, gp.ArraySpec(
                roi=gp.Roi((0, 0, 0), (200, 200, 200)),
                voxel_size=self.voxel_size,
                interpolatable=False))

    def provide(self, request):
        roi_array = request[gp.ArrayKeys.M_PRED].roi
        batch = gp.Batch()
        batch.arrays[gp.ArrayKeys.M_PRED] = gp.Array(
            self.m_pred[(roi_array / self.voxel_size).to_slices()],
            spec=gp.ArraySpec(roi=roi_array, voxel_size=self.voxel_size))
        slices = (roi_array / self.voxel_size).to_slices()
        batch.arrays[gp.ArrayKeys.D_PRED] = gp.Array(
            self.d_pred[:, slices[0], slices[1], slices[2]],
            spec=gp.ArraySpec(roi=roi_array, voxel_size=self.voxel_size))

        return batch


class TestExtractSynapses(unittest.TestCase):

    def test_output_basics(self):
        d_pred = gp.ArrayKeys.D_PRED
        m_pred = gp.ArrayKeys.M_PRED
        presyn = gp.PointsKeys.PRESYN
        postsyn = gp.PointsKeys.POSTSYN

        voxel_size = gp.Coordinate((10, 10, 10))
        size = ((200, 200, 200))
        context = 40
        shape = gp.Coordinate(size) / voxel_size
        m_predar = np.zeros(shape, dtype=np.float32)
        insidepoint = gp.Coordinate((10, 10, 10))
        outsidepoint = gp.Coordinate((15, 15, 15))
        m_predar[insidepoint] = 1
        m_predar[outsidepoint] = 1

        d_predar = np.ones((3, shape[0], shape[1], shape[2])) * 10

        outdir = tempfile.mkdtemp()

        pipeline = (
                TestSource(m_predar, d_predar,
                           voxel_size=voxel_size) +
                ExtractSynapses(
                    m_pred, d_pred, presyn, postsyn, out_dir=outdir, settings=parameters, context=context
                )
        )

        request = gp.BatchRequest()

        roi = gp.Roi((40, 40, 40), (80, 80, 80))

        request[presyn] = gp.PointsSpec(roi=roi)
        request[postsyn] = gp.PointsSpec(roi=roi)
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
        print(outdir, "outdir")
        synapsefile = os.path.join(outdir, "40", "40", "40.npz")
        with np.load(synapsefile) as data:
            data = dict(data)

        self.assertTrue(len(data['ids']) == 1)
        self.assertEqual(data['scores'][0],
                         1.0)  # Size of the cube.
        for ii in range(len(voxel_size)):
            self.assertEqual(data['positions'][0][1][ii],
                             insidepoint[ii] * voxel_size[ii])

        for ii in range(len(voxel_size)):
            self.assertEqual(data['positions'][0][0][ii],
                             insidepoint[ii] * voxel_size[ii] + 10)
        shutil.rmtree(outdir)

    def test_context(self):
        d_pred = gp.ArrayKeys.D_PRED
        m_pred = gp.ArrayKeys.M_PRED
        presyn = gp.PointsKeys.PRESYN
        postsyn = gp.PointsKeys.POSTSYN

        outdir = tempfile.mkdtemp()

        voxel_size = gp.Coordinate((10, 10, 10))
        size = ((200, 200, 200))
        # Check whether the score of the entire cube is measured, although
        # cube of borderpoint partially outside request ROI.
        context = 40
        shape = gp.Coordinate(size) / voxel_size
        m_predar = np.zeros(shape, dtype=np.float32)
        outsidepoint = gp.Coordinate((13, 13, 13))
        borderpoint = (4, 4, 4)
        m_predar[3:5, 3:5, 3:5] = 1
        m_predar[outsidepoint] = 1

        d_predar = np.ones((3, shape[0], shape[1], shape[2])) * 0

        pipeline = (
                TestSource(m_predar, d_predar,
                           voxel_size=voxel_size) +
                ExtractSynapses(
                    m_pred, d_pred, presyn, postsyn, out_dir=outdir, settings=parameters, context=context
                ) +
                gp.PrintProfilingStats()
        )

        request = gp.BatchRequest()

        roi = gp.Roi((40, 40, 40), (80, 80, 80))

        request[presyn] = gp.PointsSpec(roi=roi)
        request[postsyn] = gp.PointsSpec(roi=roi)
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        synapsefile = os.path.join(outdir, "40", "40", "40.npz")
        with np.load(synapsefile) as data:
            data = dict(data)

        self.assertTrue(len(data['ids']) == 1)
        self.assertEqual(data['scores'][0],
                         2.0 ** 3)  # Size of the cube.
        for ii in range(len(voxel_size)):
            self.assertEqual(data['positions'][0][0][ii],
                             borderpoint[ii] * voxel_size[ii])

        for ii in range(len(voxel_size)):
            self.assertEqual(data['positions'][0][1][ii],
                             borderpoint[ii] * voxel_size[ii] + 0)
        shutil.rmtree(outdir)


if __name__ == '__main__':
    unittest.main()
