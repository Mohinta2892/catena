import json
import logging
import multiprocessing as mp
import sys
import pandas as pd
import random

import daisy
import numpy as np
from pymongo import MongoClient
from scipy.spatial import KDTree
import sqlite3
from funlib.math import cantor_number

from . import database, synapse, evaluation

logger = logging.getLogger(__name__)

try:
    from lsd import local_segmentation
except ImportError:
    local_segmentation = None
    logger.warning('Could not import lsd, mapping using a local segmentation is not possible')


def get_random_links(num_samples, cursor, table='synlinks', fast=False):
    cols = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y',
            'post_z', 'scores', 'segmentid_pre', 'segmentid_post',
            'cleft_scores']
    if fast:
        com = 'SELECT {} FROM {} WHERE random() % 1000 = 0 LIMIT {};'.format(
            ','.join(cols), table, num_samples)
    else:
        com = 'SELECT {} FROM {} ORDER BY Random() LIMIT {};'.format(
            ','.join(cols), table, num_samples)

    cursor.execute(com)
    return pd.DataFrame(cursor.fetchall(), columns=cols)

class SynapseMapping(object):
    '''Maps synapses to ground truth skeletons and writes them into a
    database.
    It uses euclidean distance to cope with ambigious cases, eg. if one
    segment contains multiple skeletons, the pre/post site is mapped to
    the closest skeleton in euclidean space. If distance_upper_bound is set,
    only synapses are mapped to a skeleton if it closer than
    distance_upper_bound. Synapses that have been not mapped because of
    distance_upper_bound are marked with a skel_id of -2 (but still written
    out to database).

    Args:
        skel_db_name (``str``):
            Skeletons used for mapping.
        skel_db_host (``str``):
            Skeletons used for mapping.
        skel_db_col (``str``):
            Skeletons used for mapping.
        output_db_name (``str``):
            Mongodb name to which synapses are written out. If not provided
            syn_db_name is used.
        output_db_host (``str``)":
            If not provided, syn_db_host is used.
        output_db_col (``str``):
            If not provided, syn_db_col is used and distance_upper_bound is added.
        syndir (``str``):
            Synapses to be mapped stored in hirachical directory structure.
        syn_db_name (``str``):
            Synapses to be mapped stored in mongodb.
        syn_db_host (``str``):
            Synapses to be mapped stored in mongodb.
        syn_db_col (``str``):
            Synapses to be mapped stored in mongodb.
        gtsyn_db_name (``str``):
            If provided, those synapses are used as additional "skeletons" to
            which the synapses can be mapped to. Those nodes are ignored by
            num_skel_nodes_ignore eg. they are always used for mapping.
        gtsyn_db_host (``str``).
        gtsyn_db_col (``str``)
        seg_agglomeration_json (``str``): Jsonfile to produce a local segmentation.
        distance_upper_bound (float): If synapses are further away than
            distance_upper_bound, they are not mapped to the skeleton, although
            they are intersecting the same segment.
        num_skel_nodes_ignore (``int``): Ignore skeletons that intersect with
            number of skeletons: num_skel_nodes_ignore or less. This is used
            to account for noisy/incorrectly placed skeleton nodes, which
            should be ignored during mapping.
        multiprocess (``bool``): Whether to parallelize synapse mapping using
            the python-package multiprocessing
        mp_processes (``int``): Number of processes when multiprocess is set to
            true.
        draw_random_from_sql (``str``): sql database, from which randomly
            synapses are drawn. If this is set, syn_db_name is not used, instead,
            synapse direction vectors are drawn at random from provided database.
            Can be used to generate a "baseline" experiment. Only synapses are
            written out that have different pre and post skeleton ids
            (to reduce #of synapses).
        random_density (``float``): if draw_random_sql is set, provide the
            density in # synapses / cubic micron

    '''

    def __init__(self, skel_db_name, skel_db_host, skel_db_col,
                 output_db_name=None, output_db_host=None,
                 output_db_col=None,
                 syndir=None, syn_db_name=None, syn_db_host=None,
                 syn_db_col=None, gtsyn_db_name=None, gtsyn_db_host=None,
                 gtsyn_db_col=None,
                 seg_agglomeration_json=None,
                 distance_upper_bound=None, num_skel_nodes_ignore=0,
                 multiprocess=False, mp_processes=40,
                 draw_random_from_sql=None,
                 random_density=10):
        assert syndir is not None or syn_db_col is not None or draw_random_from_sql is not None, 'synapses have to be ' \
                                                         'provided either in syndir format or db format'

        self.skel_db_name = skel_db_name
        self.skel_db_host = skel_db_host
        self.skel_db_col = skel_db_col
        self.syn_db_name = syn_db_name
        self.syn_db_host = syn_db_host
        self.syn_db_col = syn_db_col
        self.syndir = syndir
        self.gtsyn_db_name = gtsyn_db_name
        self.gtsyn_db_host = gtsyn_db_host
        self.gtsyn_db_col = gtsyn_db_col
        self.output_db_name = output_db_name if output_db_name is not None else self.syn_db_name
        self.output_db_host = output_db_host if output_db_host is not None else self.syn_db_host
        output_db_col = output_db_col if output_db_col is not None else self.syn_db_col
        self.output_db_col = output_db_col + '_skel_{}'.format('inf' if
                                                               distance_upper_bound is None else distance_upper_bound)
        self.seg_agglomeration_json = seg_agglomeration_json
        self.distance_upper_bound = distance_upper_bound
        self.num_skel_nodes_ignore = num_skel_nodes_ignore
        self.multiprocess = multiprocess
        self.mp_processes = mp_processes
        self.skel_df = pd.DataFrame()
        self.draw_random_from_sql = draw_random_from_sql
        self.random_density = random_density

    def __match_position_to_closest_skeleton(self, position, seg_id, skel_ids):
        distances = []
        for skel_id in skel_ids:
            locations = list(self.skel_df[(self.skel_df.seg_id == seg_id) & (self.skel_df.neuron_id == skel_id)].position.apply(np.array))
            tree = KDTree(locations)
            dist = tree.query(x=np.array(position), k=1, eps=0, p=2,
                              distance_upper_bound=np.inf)[0]
            distances.append(dist)
        indexmin = np.argmin(np.array(distances))
        logger.debug('matching node to skeleton with distance: %0.2f '
                     'compared to average distance: %0.2f' % (
                         distances[indexmin], np.mean(distances)))
        if self.distance_upper_bound is not None:
            if distances[indexmin] > self.distance_upper_bound:
                logger.debug(
                    'synapse not mapped because distance {:0.2} '
                    'bigger than {:}'.format(
                        distances[indexmin], self.distance_upper_bound))
                return -2

        return skel_ids[indexmin]

    def match_synapses_to_skeleton(self, synapses):

        for ii, syn in enumerate(synapses):
            logger.debug('{}/{}'.format(ii, len(synapses)))

            # Which skeletons intersect with segmentation id ?
            skel_ids = list(np.unique(self.skel_df[self.skel_df['seg_id'] == syn.id_segm_pre]['neuron_id']))
            if self.num_skel_nodes_ignore > 0:
                for skel_id in skel_ids:
                    # With how many nodes does the skeleton intersect
                    # current segment ?
                    num_nodes = len(np.unique(self.skel_df[(self.skel_df[
                                                                'seg_id'] == syn.id_segm_pre) & (
                                                                       self.skel_df[
                                                                           'neuron_id'] == skel_id)][
                                                  'id']))
                    node_types = np.unique(self.skel_df[(self.skel_df[
                                                                'seg_id'] == syn.id_segm_pre) & (
                                                                       self.skel_df[
                                                                           'neuron_id'] == skel_id)][
                                                  'type'])
                    include_node = 'connector' in node_types or 'post_tree_node' in node_types

                    # Exclude skeletons when they have fewer numbers inside the
                    # segment than num_skel_nodes_ignore.
                    if 0 < num_nodes <= self.num_skel_nodes_ignore and not include_node:
                        logger.debug(
                            'ignoring skel id: {} syn id: {}'.format(
                                skel_id, syn.id))
                        skel_ids.remove(skel_id)

            if len(skel_ids) > 0:
                skel_ids = [
                    self.__match_position_to_closest_skeleton(syn.location_pre,
                                                              syn.id_segm_pre,
                                                              skel_ids)]
            syn.id_skel_pre = skel_ids[0] if len(skel_ids) > 0 else None

            skel_ids = list(np.unique(self.skel_df[self.skel_df['seg_id'] == syn.id_segm_post]['neuron_id']))
            for skel_id in skel_ids:
                num_nodes = len(np.unique(self.skel_df[(self.skel_df[
                                                            'seg_id'] == syn.id_segm_post) & (
                                                               self.skel_df[
                                                                   'neuron_id'] == skel_id)][
                                              'id']))
                node_types = np.unique(self.skel_df[(self.skel_df[
                                                         'seg_id'] == syn.id_segm_post) & (
                                                            self.skel_df[
                                                                'neuron_id'] == skel_id)][
                                           'type'])
                include_node = 'connector' in node_types or 'post_tree_node' in node_types

                # Exclude skeletons when they have fewer numbers inside the
                # segment than num_skel_nodes_ignore.
                if 0 < num_nodes <= self.num_skel_nodes_ignore and not include_node:
                    logger.debug(
                        'ignoring skel id: {} syn id: {}'.format(
                            skel_id, syn.id))
                    skel_ids.remove(skel_id)
            if len(skel_ids) > 0:
                skel_ids = [
                    self.__match_position_to_closest_skeleton(syn.location_post,
                                                              syn.id_segm_post,
                                                              skel_ids)]

            syn.id_skel_post = skel_ids[0] if len(skel_ids) > 0 else None
        syn_db = database.SynapseDatabase(self.output_db_name,
                                          db_host=self.output_db_host,
                                          db_col_name=self.output_db_col,
                                          mode='r+')
        if self.draw_random_from_sql is not None:
            synapses = [syn for syn in synapses if syn.id_skel_pre != syn.id_skel_post]
        syn_db.write_synapses(synapses)
        logger.info('wrote mapped synapses')

    def add_skel_ids_daisy(self, roi_core, roi_context, seg_thr,
                           seg_ids_ignore):
        """Maps synapses using a local segmentation.

        Args:
            roi_core: (``daisy.ROI``): The ROI that is used to read in the
                synapses.
            roi_context (``daisy.ROI``): The ROI that is used to read in
                skeletons and ground truth synapses, that are used for mapping.
            seg_thr (``float``): Edge score threshold used for agglomerating
                fragments to produce a local segmentation used for mapping.
            seg_ids_ignore (``list`` of ``int``):
                List of ids that are not used for mapping. Eg. all skeletons
                whose seg id are in seg_ids_ignore are removed and not used
                for mapping.

        """
        with open(self.seg_agglomeration_json) as f:
            seg_config = json.load(f)

        # Get actual segmentation ROI
        seg = daisy.open_ds(seg_config['fragments_file'],
                            seg_config['fragments_dataset'])

        # This reads in all synapses, where postsynaptic site is in ROI, but
        # it is not guaranteed, that presynaptic site is also in ROI.
        if self.draw_random_from_sql is None:
            if self.syndir is not None:
                synapses = synapse.read_synapses_in_roi(self.syndir,
                                                        roi_core)
            else:
                syn_db = database.SynapseDatabase(self.syn_db_name,
                                                  db_host=self.syn_db_host,
                                                  db_col_name=self.syn_db_col,
                                                  mode='r')
                synapses = syn_db.read_synapses(roi=roi_core)
                synapses = synapse.create_synapses_from_db(synapses)
        else:
            size_cubmicrons = roi_core.size()/1000**3
            num_of_synapses = int(size_cubmicrons*self.random_density)

            # Generate num_of_synapses postsynaptic random locations.
            # if N of possible points gets really big, this solution is not nice :(
            zlist = range(int(roi_core.get_begin()[0]/seg.voxel_size[0]), int(roi_core.get_end()[0]/seg.voxel_size[0]))
            ylist = range(int(roi_core.get_begin()[1]/seg.voxel_size[1]), int(roi_core.get_end()[1]/seg.voxel_size[1]))
            xlist = range(int(roi_core.get_begin()[2]/seg.voxel_size[2]), int(roi_core.get_end()[2]/seg.voxel_size[2]))
            coords = [[z, y, x] for x in xlist for y in ylist for z in zlist]
            post_sites = random.sample(coords, num_of_synapses)
            conn = sqlite3.connect(self.draw_random_from_sql)
            c = conn.cursor()
            links = get_random_links(num_of_synapses, c) # returns pandas dataframe
            synapses = []
            for ii, location_post in enumerate(post_sites):
                id = cantor_number(location_post)
                location_post *= np.array(seg.voxel_size)
                link = links.loc[ii]
                dir_vec = np.array((link.pre_z, link.pre_y, link.pre_x))-np.array((link.post_z, link.post_y, link.post_x))
                syn = synapse.Synapse(
                    id=np.int64(id),
                    score=link.scores,
                    location_pre=location_post+dir_vec,
                    location_post=location_post
                )
                synapses.append(syn)



        # Make sure to only look at synapses that are inside segmentation ROI.
        synapses = [syn for syn in synapses if
                    seg.roi.contains(syn.location_pre)
                    and seg.roi.contains(syn.location_post)]

        if len(synapses) == 0:
            logger.debug('no synapse in roi')
            return 0

        pre_locations = [daisy.Coordinate(syn.location_pre) for syn in synapses]
        post_locations = [daisy.Coordinate(syn.location_post) for syn in
                          synapses]
        # Compute Bounding box for pre_locations
        z_min, y_min, x_min = np.min(np.array(pre_locations), axis=0)
        z_max, y_max, x_max = np.min(np.array(pre_locations), axis=0)

        roi_big = daisy.Roi((z_min, y_min, x_min),
                            (z_max - z_min, y_max - y_min, x_max - x_min))
        roi_big = roi_big.union(roi_context)

        roi_big = roi_big.snap_to_grid(seg.voxel_size)
        roi_big = seg.roi.intersect(roi_big)

        # Load skeletons.
        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')
        nodes = gt_db.read_nodes(roi_context)
        logger.info('number of skel nodes {}'.format(len(nodes)))
        if self.gtsyn_db_name is not None:
            gt_db = database.SynapseDatabase(self.gtsyn_db_name,
                                             db_host=self.gtsyn_db_host,
                                             db_col_name=self.gtsyn_db_col,
                                             mode='r')
            gt_synapses = gt_db.read_synapses(pre_post_roi=roi_big)
            gt_synapses = pd.DataFrame(gt_synapses)
        else:
            gt_synapses = pd.DataFrame([])
        if len(nodes) == 0 and len(gt_synapses) == 0:
            logger.info('Neither skeleton nor synapse node found.')
            return 0


        logger.debug('creating a local segmentation')
        locseg = local_segmentation.LocalSegmentationExtractor(**seg_config)
        seg = locseg.get_local_segmentation(roi_big, seg_thr)

        nodes_df = pd.DataFrame(nodes)
        if len(nodes_df) > 0:
            nodes_df = nodes_df[nodes_df.apply(
                lambda row: seg.roi.contains(daisy.Coordinate(row['position'])),
                axis=1)]
            nodes_df['seg_id'] = nodes_df.apply(lambda row:
                                                seg[daisy.Coordinate(row['position'])], axis=1)


        # # Also add ground truth connectors.
        if self.gtsyn_db_name is not None:
            use_tree_node = True
            if not 'pre_node_id' in gt_synapses:
                logger.warning(
                    'No tree nodes available, assining new node ids. '
                    'Neuron nodes and synapse nodes might be counted multiple times.')
                use_tree_node = False
            if len(gt_synapses) == 0:
                logger.debug('No Ground Truth synapses')
            else:
                logger.info(
                    'number of catmaid synapses: {}'.format(len(gt_synapses)))
                pre_nodes = pd.DataFrame()
                pre_nodes['neuron_id'] = gt_synapses['pre_skel_id']
                pre_nodes['position'] = list(
                    zip(gt_synapses['pre_z'], gt_synapses['pre_y'],
                        gt_synapses['pre_x']))
                pre_nodes['type'] = 'connector'
                pre_nodes['id'] = gt_synapses.pre_node_id if use_tree_node else list(range(len(gt_synapses)))

                post_nodes = pd.DataFrame()
                post_nodes['neuron_id'] = gt_synapses['post_skel_id']
                post_nodes['position'] = list(
                    zip(gt_synapses['post_z'], gt_synapses['post_y'],
                        gt_synapses['post_x']))
                post_nodes['type'] = 'post_tree_node'
                post_nodes['id'] = gt_synapses.post_node_id if use_tree_node else list(range(len(gt_synapses), 2*len(gt_synapses)))

                syn_nodes = pd.concat([pre_nodes, post_nodes])
                syn_nodes = syn_nodes[syn_nodes.apply(
                    lambda row: seg.roi.contains(daisy.Coordinate(row['position'])),
                    axis=1)]
                syn_nodes['seg_id'] = syn_nodes.apply(lambda row:
                                                      seg[daisy.Coordinate(
                                                          row['position'])], axis=1)
                nodes_df = nodes_df.append(syn_nodes, sort=False)


        # if len(nodes_df) == 0:
        #     logger.info('Neither skeleton nor synapse node found.')
        #     return 0
        nodes_df = nodes_df[~nodes_df.seg_id.isin(seg_ids_ignore)]

        pre_ids = [seg[pre_loc] for pre_loc in pre_locations]
        post_ids = [seg[post_loc] for post_loc in post_locations]
        syn_on_skels = []
        for ii, syn in enumerate(synapses):
            pre_id = pre_ids[ii]
            post_id = post_ids[ii]

            # Test whether segment intersects with a skeleton
            skel_syn_pre = not nodes_df[nodes_df.seg_id == pre_id].empty
            skel_syn_post = not nodes_df[nodes_df.seg_id == post_id].empty
            if skel_syn_pre or skel_syn_post:
                syn.id_segm_pre = pre_id
                syn.id_segm_post = post_id
                syn_on_skels.append(syn)

        self.skel_df = nodes_df
        logger.debug(
            'matching {} synapses to skeletons, original number of synapses {}'.format(
                len(syn_on_skels), len(synapses)))
        self.match_synapses_to_skeleton(syn_on_skels)

    def add_skel_ids(self, seg_ids_ignore=[]):
        """Maps synapses to ground truth skeletons and writes them into a
        database.
        It is assumend that each ground truth neuron and each pre and
        postsynaptic site has already a segmentation ID assigned.

        Args:
            seg_ids_ignore (``list`` of ``int``):
                List of ids that are not used for mapping. Eg. all skeletons
                whose seg id are in seg_ids_ignore are removed and not used
                for mapping.
        """

        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')

        pred_db = database.SynapseDatabase(self.syn_db_name,
                                           db_host=self.syn_db_host,
                                           db_col_name=self.syn_db_col,
                                           mode='r')

        nodes = gt_db.read_nodes()
        nodes_df = pd.DataFrame(nodes)

        # Also add ground truth connectors.
        if self.gtsyn_db_name is not None:
            gt_db = database.SynapseDatabase(self.gtsyn_db_name,
                                             db_host=self.gtsyn_db_host,
                                             db_col_name=self.gtsyn_db_col,
                                             mode='r')
            gt_synapses = gt_db.read_synapses()
            gt_synapses = pd.DataFrame(gt_synapses)
            use_tree_node = True
            if not 'pre_node_id' in gt_synapses:
                logger.warning(
                    'No tree nodes available, assining new node ids. '
                    'Neuron nodes and synapse nodes might be counted multiple times.')
                use_tree_node = False

            if len(gt_synapses) == 0:
                logger.debug('No Ground Truth synapses')
            else:
                logger.info(
                    'number of catmaid synapses: {}'.format(len(gt_synapses)))
                pre_nodes = pd.DataFrame()
                pre_nodes['neuron_id'] = gt_synapses['pre_skel_id']
                pre_nodes['position'] = list(
                    zip(gt_synapses['pre_z'], gt_synapses['pre_y'],
                        gt_synapses['pre_x']))
                pre_nodes['type'] = 'connector'
                pre_nodes['id'] = gt_synapses.pre_node_id if use_tree_node else list(range(len(gt_synapses)))
                pre_nodes['seg_id'] = gt_synapses.pre_seg_id

                post_nodes = pd.DataFrame()
                post_nodes['neuron_id'] = gt_synapses['post_skel_id']
                post_nodes['position'] = list(
                    zip(gt_synapses['post_z'], gt_synapses['post_y'],
                        gt_synapses['post_x']))
                post_nodes['type'] = 'post_tree_node'
                post_nodes['id'] = gt_synapses.post_node_id if use_tree_node else list(range(len(gt_synapses), 2*len(gt_synapses)))
                post_nodes['seg_id'] = gt_synapses.post_seg_id

                syn_nodes = pd.concat([pre_nodes, post_nodes])
                nodes_df = nodes_df.append(syn_nodes, sort=False)


        if len(nodes_df) == 0:
            logger.info('Neither skeleton nor synapse node found.')
            return 0
        nodes_df = nodes_df[~nodes_df.seg_id.isin(seg_ids_ignore)]
        self.skel_df = nodes_df

        seg_ids = list(np.unique(nodes_df.seg_id))
        seg_ids = [int(id) for id in seg_ids]
        synapses = pred_db.synapses.find({'$or': [
            {'pre_seg_id': {'$in': seg_ids}},
            {'post_seg_id': {'$in': seg_ids}},
        ]})
        synapses = synapse.create_synapses_from_db(synapses)

        logger.info('found {} synapses '.format(len(synapses)))
        logger.info('Overwriting {}/{}/{}'.format(self.output_db_name,
                                                  self.output_db_host,
                                                  self.output_db_col))
        syn_db = database.SynapseDatabase(self.output_db_name,
                                          db_host=self.output_db_host,
                                          db_col_name=self.output_db_col,
                                          mode='w')
        batch_size = 100
        args = []
        for ii in range(0, len(synapses), batch_size):
            if self.multiprocess:
                args.append(synapses[ii:ii + batch_size])
            else:
                self.match_synapses_to_skeleton(synapses[ii:ii + batch_size])

        if self.multiprocess:
            pool = mp.Pool(self.mp_processes)
            pool.map(self.match_synapses_to_skeleton, args)
            pool.close()
            pool.join()