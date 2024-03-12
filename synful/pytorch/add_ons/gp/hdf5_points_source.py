import copy
import logging

import numpy as np
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.nodes.batch_provider import BatchProvider
# from .points import PointsKeys, Points, PreSynPoint, PostSynPoint
from .points_spec import PointsSpec
from gunpowder.graph_spec import GraphSpec
from gunpowder.graph import GraphKey, GraphKeys, Graph
from gunpowder.profiling import Timing
from .prepost_points_graphkey import PostSynPoint, PreSynPoint  # dependency Node from Graph
from .points import Points  # dependency Freezable

logger = logging.getLogger(__name__)


class Hdf5PointsSource(BatchProvider):
    """
    An HDF5 data source for :class:``Graphs`` (formerly :class:``Points``).
    Currently only supports a
    specific case where points represent pre- and post-synaptic markers.

    Args:

        filename (string): The HDF5 file.

        datasets (dict): Dictionary of :class:``PointsKey`` -> dataset names
            that this source offers.

        rois (dict): Dictionary of :class:``PointsKey`` -> :class:``Roi`` to
            set the ROI for each point set provided by this source.

        kind (string): allowed arguments are synapse, presyn, postsyn. If
            synaptic partners should be loaded, choose: synapse -->
            provide two pointskeys: PRESYN and POSTSYN. If only pre or postsyn
            should be loaded, without the respective partner, choose presyn or
            postsyn --> only provide one pointkey.

   """

    def __init__(
            self,
            filename,
            datasets,
            rois=None,
            kind='synapse'):

        self.filename = filename
        self.datasets = datasets
        self.rois = rois
        self.kind = kind  # partner, presyn or postsyn
        self.ndims = None
        assert kind == 'synapse' or kind == 'presyn' or kind == 'postsyn', \
            "option -kind- set to {}, Hdf5PointsSource implemented only " \
            "for synapse, presyn or postsyn".format(kind)
        if kind == 'synapse':
            assert len(datasets) == 2, "option kind set to synapse, " \
                                       "provide GraphKeys (formerly PointsKeys) for Pre- and Postsynapse"
        else:
            assert len(datasets) == 1

    def setup(self):

        hdf_file = h5py.File(self.filename, 'r')

        for (points_key, ds_name) in self.datasets.items():

            if ds_name not in hdf_file:
                raise RuntimeError("%s not in %s" % (ds_name, self.filename))

            # spec = PointsSpec()
            spec = GraphSpec()
            if self.rois is not None:
                if points_key in self.rois:
                    spec.roi = self.rois[points_key]

            self.provides(points_key, spec)

        hdf_file.close()

    def provide(self, request):

        timing_process = Timing(self)
        timing_process.start()

        batch = Batch()

        with h5py.File(self.filename, 'r') as hdf_file:

            # if pre and postsynaptic locations required, their id
            # SynapseLocation dictionaries should be created together s.t. ids
            # are unique and allow to find partner locations

            # if PointsKeys.PRESYN in request.points_specs or PointsKeys.POSTSYN in request.points_specs:
            if GraphKeys.PRESYN in request.graph_specs or GraphKeys.POSTSYN in request.graph_specs:
                assert self.kind == 'synapse'
                # If only PRESYN or POSTSYN requested, assume PRESYN ROI = POSTSYN ROI.
                # pre_key = PointsKeys.PRESYN if PointsKeys.PRESYN in request.points_specs else PointsKeys.POSTSYN
                pre_key = GraphKeys.PRESYN if GraphKeys.PRESYN in request.graph_specs else GraphKeys.POSTSYN
                # post_key = PointsKeys.POSTSYN if PointsKeys.POSTSYN in request.points_specs else PointsKeys.PRESYN
                post_key = GraphKeys.POSTSYN if GraphKeys.POSTSYN in request.graph_specs else GraphKeys.PRESYN
                # presyn_points, postsyn_points = self.__get_syn_points(
                #     pre_roi=request.points_specs[pre_key].roi,
                #     post_roi=request.points_specs[post_key].roi,
                #     syn_file=hdf_file)
                presyn_points, postsyn_points = self.__get_syn_points(
                    pre_roi=request.graph_specs[pre_key].roi,
                    post_roi=request.graph_specs[post_key].roi,
                    syn_file=hdf_file)
                # points = {
                #     PointsKeys.PRESYN: presyn_points,
                #     PointsKeys.POSTSYN: postsyn_points}
                points = {
                    GraphKeys.PRESYN: presyn_points,
                    GraphKeys.POSTSYN: postsyn_points}
            else:
                assert self.kind == 'presyn' or self.kind == 'postsyn'
                synkey = list(self.datasets.items())[0][0]  # only key of dic.
                presyn_points, postsyn_points = self.__get_syn_points(
                    # pre_roi=request.points_specs[synkey].roi,
                    pre_roi=request.graph_specs[synkey].roi,
                    # post_roi=request.points_specs[synkey].roi,
                    post_roi=request.graph_specs[synkey].roi,
                    syn_file=hdf_file)
                points = {
                    synkey: presyn_points if self.kind == 'presyn' else postsyn_points
                }

            # for (points_key, request_spec) in request.points_specs.items():
            for (points_key, request_spec) in request.graph_specs.items():
                logger.debug("Reading %s in %s...", points_key, request_spec.roi)
                points_spec = self.spec[points_key].copy()
                points_spec.roi = request_spec.roi
                # logger.debug("Number of points len()".format(len(points[points_key])))
                logger.debug("Number of points len()".format(len(points[points_key])))
                # batch.points[points_key] = Points(data=points[points_key], spec=points_spec)
                # a graph with nodes and no edges, the points_spec is extracted above
                nodes = list(points[points_key].values()) # this is how BatchProvider expects it
                batch.graphs[points_key] = Graph(nodes=nodes, edges=(), spec=points_spec)

        timing_process.stop()
        batch.profiling_stats.add(timing_process)

        return batch

    def __get_syn_points(self, pre_roi, post_roi, syn_file):
        presyn_points_dict, postsyn_points_dict = {}, {}
        annotation_ids = syn_file['annotations/ids'][:]
        locs = syn_file['annotations/locations'][:]
        if 'offset' in syn_file['annotations'].attrs:
            offset = np.array(syn_file['annotations'].attrs['offset'])
            logger.debug("Retrieving offset")
        else:
            offset = None
            logger.debug('No offset')
        syn_id = 0
        for pre, post in list(
                syn_file['annotations/presynaptic_site/partners'][:]):
            pre_index = int(np.where(pre == annotation_ids)[0][0])
            post_index = int(np.where(post == annotation_ids)[0][0])
            pre_site = locs[pre_index]
            post_site = locs[post_index]
            if offset is not None:
                pre_site += offset
                post_site += offset

            if pre_roi.contains(Coordinate(pre_site)):
                syn_point = PreSynPoint(location=pre_site,
                                        location_id=pre_index,
                                        synapse_id=syn_id,
                                        partner_ids=[post_index])
                presyn_points_dict[pre_index] = copy.deepcopy(syn_point)
            if post_roi.contains(Coordinate(post_site)):
                syn_point = PostSynPoint(location=post_site,
                                         location_id=post_index,
                                         synapse_id=syn_id,
                                         partner_ids=[pre_index])
                postsyn_points_dict[post_index] = copy.deepcopy(syn_point)
            if pre_roi.contains(Coordinate(pre_site)) or post_roi.contains(Coordinate(post_site)):
                syn_id += 1

        return presyn_points_dict, postsyn_points_dict

    def __repr__(self):

        return self.filename
