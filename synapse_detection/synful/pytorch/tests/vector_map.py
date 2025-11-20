from provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    Coordinate,
    GraphSpec,
    Array,
    ArrayKey,
    ArrayKeys,
    ArraySpec,
    RasterizeGraph,
    RasterizationSettings,
    build,
    RandomLocation,
    Hdf5Source,
    MergeProvider,
    Normalize
)
from gunpowder.graph import GraphKeys, GraphKey, Graph, Node, Edge
from synful.add_ons.gp import AddPartnerVectorMap, Hdf5PointsSource

import numpy as np
import math
from random import randint
import matplotlib.pyplot as plt
import os

# ensure that only the parent folder contains a backward slash at the beginning, otherwise path.join to
# prevent path.join from discarding the base folders
home = '/media/samia/DATA/ark'
# CREMI specific, download data from: www.cremi.org
data_dir = os.path.join(home, 'dan-samia/lsd/')

data_dir_syn = data_dir
print(data_dir_syn)

samples = [
    'funke/cremi/training/3D/padded/hdf/sample_A_padded_20160501',
    # 'funke/cremi/training/3D/padded/hdf/sample_B_padded_20160501',
    # 'funke/cremi/training/3D/padded/hdf/sample_C_padded_20160501'
]
# padded
cremi_roi = Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))


# cropped
# cremi_roi = Roi(np.array((0, 0, 0)), np.array((5000, 5000, 5000)))

def create_source(sample, raw,
                  presyn, postsyn, dummypostsyn,
                  parameter,
                  gt_neurons
                  ):
    data_sources = tuple(
        (
            Hdf5PointsSource(
                os.path.join(data_dir_syn, sample + '.hdf'),
                datasets={presyn: 'annotations',
                          postsyn: 'annotations'},
                rois={
                    presyn: cremi_roi,
                    postsyn: cremi_roi
                }
            ),
            Hdf5PointsSource(
                os.path.join(data_dir_syn, sample + '.hdf'),
                datasets={
                    dummypostsyn: 'annotations'},
                rois={
                    # presyn: cremi_roi,
                    dummypostsyn: cremi_roi
                },
                kind='postsyn'
            ),
            Hdf5Source(
                os.path.join(data_dir_syn, sample + '.hdf'),
                datasets={
                    raw: 'volumes/raw',
                    gt_neurons: 'volumes/labels/neuron_ids',  # shape (125, 1250, 1250) in zyx
                },
                array_specs={
                    raw: ArraySpec(interpolatable=True),
                    gt_neurons: ArraySpec(interpolatable=False),
                }
            )
        )
    )
    source_pip = data_sources + MergeProvider() + Normalize(
        raw) + RandomLocation(ensure_nonempty=dummypostsyn,
                              p_nonempty=parameter['reject_probability'])
    return source_pip


class GraphTestSource3D(BatchProvider):
    def __init__(self):
        self.voxel_size = Coordinate((40, 4, 4))

        self.nodes = [
            # corners
            Node(id=1, location=np.array((-200, -200, -200))),
            Node(id=2, location=np.array((-200, -200, 199))),
            Node(id=3, location=np.array((-200, 199, -200))),
            Node(id=4, location=np.array((-200, 199, 199))),
            Node(id=5, location=np.array((199, -200, -200))),
            Node(id=6, location=np.array((199, -200, 199))),
            Node(id=7, location=np.array((199, 199, -200))),
            Node(id=8, location=np.array((199, 199, 199))),
            # center
            Node(id=9, location=np.array((0, 0, 0))),
            Node(id=10, location=np.array((-1, -1, -1))),
        ]

        self.graph_spec = GraphSpec(roi=Roi((-100, -100, -100), (300, 300, 300)))
        self.array_spec = ArraySpec(
            roi=Roi((-200, -200, -200), (400, 400, 400)), voxel_size=self.voxel_size,
            dtype=np.uint64
        )

        self.graph = Graph(self.nodes, [], self.graph_spec)

    def setup(self):
        self.provides(
            GraphKeys.TEST_GRAPH,
            self.graph_spec,
        )

        self.provides(
            ArrayKeys.GT_LABELS,
            self.array_spec,
        )

    def provide(self, request):
        batch = Batch()

        graph_roi = request[GraphKeys.TEST_GRAPH].roi

        batch.graphs[GraphKeys.TEST_GRAPH] = self.graph.crop(graph_roi).trim(graph_roi)

        roi_array = request[ArrayKeys.GT_LABELS].roi

        image = np.ones(roi_array.shape / self.voxel_size, dtype=np.uint64)
        # label half of GT_LABELS differently
        depth = image.shape[0]
        image[0: depth // 2] = 2

        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.GT_LABELS] = Array(image, spec=spec)

        return batch


class TestRasterizePoints(ProviderTest):
    def test_3d(self):
        GraphKey("TEST_GRAPH")
        ArrayKey("RASTERIZED")

        pipeline = GraphTestSource3D() + RasterizeGraph(
            GraphKeys.TEST_GRAPH,
            ArrayKeys.RASTERIZED,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(radius=1, fg_value=0, bg_value=1, mask=ArrayKeys.GT_LABELS),
        ) + AddPartnerVectorMap()

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data
            self.assertEqual(rasterized[0, 0, 0], 0)
            self.assertEqual(rasterized[2, 20, 20], 1)
            self.assertEqual(rasterized[4, 49, 49], 0)

# if __name__ == '__main__':
#     dummypostsyn = GraphKey('DUMMYPOSTSYN')
#     postsyn = GraphKey('POSTSYN')
#     presyn = GraphKey('PRESYN')
#     trg_context = 140  # AddPartnerVectorMap context in nm - pre-post distance
#     raw = ArrayKey('RAW')
#     gt_neurons = ArrayKey('GT_NEURONS')
#     gt_post_indicator = ArrayKey('GT_POST_INDICATOR')
#     post_loss_weight = ArrayKey('POST_LOSS_WEIGHT')
#
#     pred_post_indicator = ArrayKey('PRED_POST_INDICATOR')
#     pred_post_indicator_sigmoid = ArrayKey('PRED_POST_INDICATOR_SIGMOID')
#
#     grad_syn_indicator = ArrayKey('GRAD_SYN_INDICATOR')
#     vectors_mask = ArrayKey('VECTORS_MASK')
#     gt_postpre_vectors = ArrayKey('GT_POSTPRE_VECTORS')
