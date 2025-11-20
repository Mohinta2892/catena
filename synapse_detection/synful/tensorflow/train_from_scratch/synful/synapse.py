import copy
import glob
import logging
import os
from itertools import product, starmap
import scipy
import json

import h5py
import numpy as np
from scipy.sparse.csgraph import csgraph_from_dense, connected_components
import neuroglancer

logger = logging.getLogger(__name__)

try:
    import navis
except ImportError:
    logger.warning('Could not import navis, functions with CATMAID skeletons '
                   'will not work')


class Synapse(object):
    """Represents a single synapse.
    """

    def __init__(self, id=None, id_segm_pre=None, id_segm_post=None,
                 location_pre=None, location_post=None, score=None,
                 id_skel_pre=None, id_skel_post=None, node_id_pre=None,
                 node_id_post=None):
        self.id = id
        self.id_segm_pre = id_segm_pre
        self.id_segm_post = id_segm_post
        self.location_pre = location_pre
        self.location_post = location_post
        self.id_skel_pre = id_skel_pre
        self.id_skel_post = id_skel_post
        self.node_id_pre = node_id_pre
        self.node_id_post = node_id_post
        self.score = score

    def __repr__(self):
        output_str = 'id: %s, seg_ids: [%s, %s], skel_ids: [%s, %s], score: %s' % (
            str(self.id),
            str(self.id_segm_pre),
            str(self.id_segm_post),
            str(self.id_skel_pre),
            str(self.id_skel_post),
            '{:0.3f}'.format(self.score) if self.score is not None else None)
        return output_str

def create_synapses_from_db(synapses_dic):
    synapses = []
    for syn_dic in synapses_dic:
        if 'post_z' in syn_dic:
            post_loc = np.array(
                (syn_dic['post_z'], syn_dic['post_y'], syn_dic['post_x']))
        else:
            post_loc = None
        if 'pre_z' in syn_dic:
            pre_loc = np.array(
                (syn_dic['pre_z'], syn_dic['pre_y'], syn_dic['pre_x']))
        else:
            pre_loc = None
        syn = Synapse(
            id=syn_dic['id'],
            location_pre=pre_loc,
            location_post=post_loc,
        )
        syn.id_skel_pre = syn_dic.get('pre_skel_id', None)
        syn.id_skel_post = syn_dic.get('post_skel_id', None)
        syn.id_segm_pre = syn_dic.get('pre_seg_id', None)
        syn.id_segm_post = syn_dic.get('post_seg_id', None)
        syn.node_id_pre = syn_dic.get('pre_node_id', None)
        syn.node_id_post = syn_dic.get('post_node_id', None)
        syn.score = syn_dic.get('score', None)

        synapses.append(syn)
    return synapses


def create_synapses(sources, targets, scores=None):
    """Creates a list of synapses.

    Args:
        sources (list): List with source positions.
        targets (list): List with target positions.

    """
    assert len(sources) == len(targets)
    synapses = []
    score = None
    counter = 0
    for presite, postsite in zip(*[sources, targets]):
        if scores is not None:
            score = scores[counter]
        synapses.append(Synapse(location_pre=presite,
                                location_post=postsite,
                                score=score))
        counter += 1
    return synapses


def __ndrange(start, stop=None, step=None):
    if stop is None:
        stop = start
        start = (0,)*len(stop)

    if step is None:
        step = (1,)*len(stop)

    assert len(start) == len(stop) == len(step)

    indeces = []
    for index in product(*starmap(range, zip(start, stop, step))):
        indeces.append(index)
    return indeces


def __get_chunk_size(directory):
    offsets = [np.int64(f.name) for f in os.scandir(directory) if f.is_dir()]
    offsets.sort()
    zchunksize = offsets[1]

    offsets = [np.int64(f.name) for f in
               os.scandir(os.path.join(directory, '0')) if f.is_dir()]
    offsets.sort()
    ychunksize = offsets[1]

    offsets = glob.glob(os.path.join(directory, '0', '0', '*.npz'))
    offsets = [np.int64(os.path.splitext(os.path.basename(f))[0]) for f in
               offsets]
    offsets.sort()
    xchunksize = offsets[1]

    return (zchunksize, ychunksize, xchunksize)


def read_synapses_in_roi(directory, roi, chunk_size=None, score_thr=-1):
    """ Reads synapses from a npz files stored in x, y, z dir structure.

    Args:
        directory (str): Input directory where synapses are stored
        roi (daisy.Roi): Synapses are only read in this ROI.
        chunk_size (tuple): Size of chunks / blocks. If not provided, it will
        be tried to infer this value from the dir structure itself.

    Returns:
        synapses (list): Returns list of synapse.Synapse

    """

    if chunk_size is None:
        attrfilename = os.path.join(directory, 'synfulattrs.json')
        if os.path.exists(attrfilename):
            with open(attrfilename) as f:
                config = json.load(f)
                chunk_size = config.get('chunk_size', None)
        if chunk_size is None:
            chunk_size = __get_chunk_size(directory)
    adjusted_roi = roi.snap_to_grid(chunk_size)
    blocks = __ndrange(adjusted_roi.get_begin(), adjusted_roi.get_end(),
                     chunk_size)
    logger.debug('loading from {} block(s)'.format(len(blocks)))
    synapses = []
    block_counter = 0
    for z, y, x in blocks:
        filename = os.path.join(directory, str(z), str(y), '{}.npz'.format(x))
        block_counter += 1
        if block_counter%5000==0:
            logger.debug('loading {}/{}, number of synapses: {}'.format(block_counter, len(blocks), len(synapses)))
        if os.path.exists(filename):
            locations = np.load(filename)['positions']
            scores = np.load(filename)['scores']
            ids = np.load(filename)['ids']
            for ii, id in enumerate(ids):
                syn = Synapse(id=id, score=scores[ii],
                              location_pre=locations[ii, 0],
                              location_post=locations[ii, 1])
                if roi.contains(syn.location_post) and syn.score > score_thr:
                    synapses.append(syn)

    return synapses


def write_synapses_into_cremiformat(synapses, filename, offset=None,
                                    overwrite=False):
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    for synapse in synapses:
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([id_nr, id_nr + 1])
        partners.extend([np.array((id_nr, id_nr + 1))])
        assert synapse.location_pre is not None and synapse.location_post is not None
        locations.extend(
            [np.array(synapse.location_pre), np.array(synapse.location_post)])
        id_nr += 2
    if overwrite:
        h5_file = h5py.File(filename, 'w')
    else:
        h5_file = h5py.File(filename, 'a')
    dset = h5_file.create_dataset('annotations/ids', data=ids,
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/locations',
                                  data=np.stack(locations, axis=0).astype(np.float32),
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                  data=np.stack(partners, axis=0).astype(np.uint32),
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/types', data=np.array(types, dtype='S'),
                                  compression='gzip')

    if offset is not None:
        h5_file['annotations'].attrs['offset'] = offset
    h5_file.close()
    logger.debug('File written to {}'.format(filename))


def __find_redundant_synapses(synapses, dist_threshold, id_type,
                              skeleton=None, ignore_ids=[]):
    pair_to_syns = {}
    for syn in synapses:
        if id_type == 'seg':
            pair = (syn.id_segm_pre, syn.id_segm_post)
        elif id_type == 'skel':
            pair = (syn.id_skel_pre, syn.id_skel_post)
        else:
            raise Exception('id_type {} not known'.format(id_type))
        if None in pair:
            continue
        ignore_pair = ignore_ids.count(pair[0]) > 0 or \
                      ignore_ids.count(pair[1]) > 0

        if not ignore_pair:
            pair_to_syns.setdefault(pair, [])
            pair_to_syns[pair].append(syn)

    clusters = []
    for pair, syns in pair_to_syns.items():
        if len(syns) > 1:
            # --> multiple synapses have same pre_id and post_id
            clustered_syns = __find_cc_of_synapses(syns, dist_threshold,
                                                   skeleton=skeleton)
            if len(clustered_syns) > 0:
                clusters.extend(clustered_syns)
    return clusters

def get_closest_treenode_ids(points, skeleton):
    dist = scipy.spatial.distance.cdist(np.array(skeleton.nodes[['z', 'y', 'x']])-np.array((40, 0, 0)), points)
    closest_node_ics = np.argmin(dist, axis=0)
    tree_node_ids = [skeleton.nodes['treenode_id'][index] for index in closest_node_ics]
    return tree_node_ids



def __find_cc_of_synapses(synapses, dist_threshold, skeleton=None):
    points = np.array([syn.location_post for syn in synapses])
    if skeleton is None:
        dists = np.sqrt(
            ((points.reshape(-1, 1, 3) - points.reshape(1, -1, 3)) ** 2).sum(
                axis=2))
    else:
        tree_node_ids = get_closest_treenode_ids(points, skeleton)
        dists = np.zeros((len(synapses), len(synapses)))
        for i, node_a in enumerate(tree_node_ids):
            for j, node_b in enumerate(tree_node_ids):
                dist = navis.graph_utils.dist_between(skeleton, node_a, node_b)
                dists[i,j] = dist if dist != 0 else -1

    # it is a symmetric matrix, remove redundancy
    dists *= np.tri(*dists.shape)

    dists[dists > dist_threshold] = np.NaN
    dists[dists == 0] = np.NaN # half of matrix set to np.nan to account for redundancy
    dists[dists == -1] = 0 # Those distances that are actually 0, were marked with -1
    sparsematrix = csgraph_from_dense(dists, null_value=np.NAN)
    num_cc, labels = connected_components(sparsematrix,
                                                               directed=False)
    clusters_of_indeces = []
    for label in np.unique(labels):
        clusters_of_indeces.append(list(np.where(labels == label)[0]))
    clustered_synapses = []
    for cluster in clusters_of_indeces:
        if len(cluster) > 1:
            cluster = [synapses[ind] for ind in cluster]
            clustered_synapses.append(cluster)
    return clustered_synapses


def cluster_synapses(synapses, dist_threshold, fuse_strategy='mean',
                     id_type='seg', skeleton=None, ignore_ids=[]):
    """ Match synapses with same seg ids in close euclidean distance or geodesic
    distance.

    Args:
        synapses (list)): List of synapse.Synapse.
        dist_threshold (float): Threshold for finding connected components.
        id_type (str): Whether to use seg_id or skel_id to find clusters.
        Defaults to seg. Possible values are seg or skel.
        skeleton (int): CATMAID skeleton id, if set, clustering is performed
        not based on euclidean distance, but on the distance that is calculated
        along the arbor (called geodesic distance).

    Returns:
        synapses (list): Returns list of synapse.Synapse with redundant synapses
        fused to single synapse.

        ids (list): Returns a  list of synapse ids, that have been removed from list

    """
    clusters = __find_redundant_synapses(synapses, dist_threshold,
                                         id_type=id_type,
                                         skeleton=skeleton,
                                         ignore_ids=ignore_ids)
    id_to_synapses = {}
    for syn in synapses:
        assert syn.id is not None
        id_to_synapses[syn.id] = syn

    all_removed_ids = []
    for cluster in clusters:
        if fuse_strategy == 'mean':
            logger.debug('Fuse_strategy mean chosen, not guaranteed that '
                         'underlying connectivity stays the same')
            new_loc_post = np.mean(np.array([syn.location_post for syn in cluster]),
                                   axis=0)
            new_loc_pre = np.mean(np.array([syn.location_pre for syn in cluster]),
                                  axis=0)
            new_id = cluster[0].id
            ids_to_remove = [syn.id for syn in cluster[1:]]
            all_removed_ids.extend(ids_to_remove)
            new_syn = copy.copy(cluster[0])
            new_syn.location_post = new_loc_post
            new_syn.location_pre = new_loc_pre
            id_to_synapses[new_id] = new_syn
        elif fuse_strategy == 'max_score':
            scores = [syn.score for syn in cluster]
            max_index = np.argmax(scores)
            ids_to_remove = [syn.id for syn in cluster]
            del ids_to_remove[max_index]
            all_removed_ids.extend(ids_to_remove)
        else:
            raise RuntimeError('fuse_strategy {} not known'.format(fuse_strategy))

        for id in ids_to_remove:
            del id_to_synapses[id]


    return list(id_to_synapses.values()), all_removed_ids


def visualize_synapses_in_neuroglancer(s, synapses, score_thr=-1, radius=30,
                 show_ellipsoid_annotation=False, name='', color='#00ff00'):
    pre_sites = []
    post_sites = []
    connectors = []
    below_score = 0
    neuro_id = 0
    for syn in synapses:
        if syn.score is None:
            add_synapse = True
        else:
            add_synapse = syn.score > score_thr
        if not add_synapse:
            below_score += 1
        else:
            pre_site = np.flip(syn.location_pre)
            post_site = np.flip(syn.location_post)
            description = f"id: {syn.id}, pre_seg: {syn.id_segm_pre}, post_seg: {syn.id_segm_post}, score: {syn.score}"
            pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
                                                              radii=(
                                                                  radius,
                                                                  radius,
                                                                  radius),
                                                              id=neuro_id + 1))
            post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                               radii=(
                                                                   radius,
                                                                   radius,
                                                                   radius),
                                                               id=neuro_id + 2))
            connectors.append(
                neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                            id=neuro_id + 3,
                                            description=description))
            neuro_id += 3

    s.layers['connectors_{}'.format(name)] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color=color,
        annotations=connectors,
    )
    if show_ellipsoid_annotation:
        s.layers['pre_sites'] = neuroglancer.AnnotationLayer(
            voxel_size=(1, 1, 1),
            filter_by_segmentation=False,
            annotation_color='#00ff00',
            annotations=pre_sites,
        )
        s.layers['post_sites'] = neuroglancer.AnnotationLayer(
            voxel_size=(1, 1, 1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=post_sites,
        )
    print(
        'filtered out {}/{} of synapses'.format(below_score,
                                                len(synapses)))
    print('displaying {} synapses'.format(len(post_sites)))
    return synapses
