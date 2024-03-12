import logging
import operator
import sqlite3

import daisy
import numpy as np
import pandas as pd
import time
from scipy import ndimage
from scipy.spatial import KDTree
from skimage import measure
from sklearn.cluster import KMeans

from . import synapse

logger = logging.getLogger(__name__)


def dda_round(x):
    return (x + 0.5).astype(int)


class DDA3:
    def __init__(self, start, end, scaling=np.array([1, 1, 1])):
        assert np.array_equal(start - np.floor(start), np.zeros(len(start)))
        assert np.array_equal(end - np.floor(end), np.zeros(len(end)))

        self.start = (start * scaling).astype(float)
        self.end = (end * scaling).astype(float)
        self.line = [dda_round(self.start)]

        self.max_direction, self.max_length = max(
            enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
        self.dv = (
                              self.end - self.start) / self.max_length if self.max_direction != 0 else np.array(
            [0] * len(self.start))

    def draw(self):
        for step in range(int(self.max_length)):
            self.line.append(dda_round((step + 1) * self.dv + self.start))

        # assert (np.all(self.line[-1] == self.end))

        for n in range(len(self.line) - 1):
            assert (np.linalg.norm(self.line[n + 1] - self.line[n]) <= np.sqrt(
                3))

        return self.line


def get_links_in_roi(roi, cursor, table='synlinks',
                     filter_autapses=False, score_thr=0):
    cols = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y',
            'post_z', 'scores', 'segmentid_pre', 'segmentid_post',
            'cleft_scores']

    z1, y1, x1 = roi.get_offset()
    z2, y2, x2 = roi.get_end()

    condition_z = '(pre_z >= {} AND pre_z <= {})'.format(z1, z2)
    condition_y = '(pre_y >= {} AND pre_y <= {})'.format(y1, y2)
    condition_x = '(pre_x >= {} AND pre_x <= {})'.format(x1, x2)

    query = 'SELECT {} from {} where {} AND {} AND {};'.format(','.join(cols),
                                                               table,
                                                               condition_z,
                                                               condition_y,
                                                               condition_x)
    print(query)
    cursor.execute(query)
    pre_links = cursor.fetchall()

    links = pd.DataFrame.from_records(pre_links, columns=cols)
    if filter_autapses:
        links = links[links.segmentid_pre != links.segmentid_post]
    if score_thr > 0:
        links = links[links.scores >= score_thr]

    return links


def cc_clefts(input, cc_thr=130):
    binarized = input >= cc_thr
    cc, num_labels = ndimage.label(binarized)
    logger.debug(f'Number of labels {num_labels}')
    regions = measure.regionprops(cc)
    cleft_id_reg = {}
    for reg in regions:
        label_id = reg['label']
        cleft_id_reg[label_id] = reg

    return cc, cleft_id_reg


def intersect_clefts_and_synapses(cc, links, roi, voxel_size, statsdata=None):
    '''

    :param cc: thresholded synaptic clefts
    :param links: pandas.Dataframe synapses
    :param roi:
    :param voxel_size:
    :return: only synapses which are fully contained in roi are used and returned.
    '''

    offset = roi.get_begin()
    for ind, link in links.iterrows():
        pos_pre = (link.pre_z, link.pre_y, link.pre_x)
        pos_post = (link.post_z, link.post_y, link.post_x)
        if roi.contains(pos_pre) and roi.contains(pos_post):
            pos_u = (np.array(pos_pre) - offset) / voxel_size
            pos_v = (np.array(pos_post) - offset) / voxel_size
            dda = DDA3(pos_u, pos_v, scaling=np.array((1, 1, 1)))
            coord_on_line = dda.draw()
            res = list(cc[list(zip(*coord_on_line))[0],
                          list(zip(*coord_on_line))[1],
                          list(zip(*coord_on_line))[2]])
            cluster_ids = list(np.unique(res))
            if statsdata is not None:
                cleft_score = np.max(
                    list(statsdata[list(zip(*coord_on_line))[0],
                                   list(zip(*coord_on_line))[1],
                                   list(zip(*coord_on_line))[2]]))
                links.loc[ind, 'cleft_score_cor'] = cleft_score

            # Majority vote
            if 0 in cluster_ids:
                cluster_ids.remove(0)
            if len(cluster_ids) > 1:
                max_oc = 0
                for cluster_id in cluster_ids:
                    oc = res.count(cluster_id)
                    if oc > max_oc:
                        winner = cluster_id
            if len(cluster_ids) == 1:
                winner = int(cluster_ids[0])
            if len(cluster_ids) == 0:
                winner = 0
            links.loc[ind, 'cleft_id'] = int(winner)
        else:
            links.loc[ind, 'cleft_id'] = -1
    links = links[links.cleft_id != -1].copy()
    return links


def add_connector_loc(df, cleft_id_reg, roi, voxel_size, split_above=None):
    voxel_size = np.array(voxel_size)
    # For each cluster, find a new loc.
    clusters = list(np.unique(df.cleft_id))

    for cluster in clusters:
        if cluster is not None and cluster != 0:
            area = cleft_id_reg[int(cluster)]['area']
            cleft_df = df[df.cleft_id == cluster]
            centroid = cleft_id_reg[int(cluster)]['centroid']
            centroid = np.array(centroid * voxel_size + roi.get_begin())
            pre_locs = list(zip(cleft_df.pre_z, cleft_df.pre_y, cleft_df.pre_x))
            tree = KDTree(pre_locs)
            dist, ind = tree.query(x=np.array(centroid), k=1, eps=0, p=2,
                                   distance_upper_bound=np.inf)
            cluster_loc = pre_locs[ind]
            cluster_loc = ((cluster_loc / voxel_size) * voxel_size).astype(
                np.int)
            df.loc[cleft_df.index, 'con_z'] = cluster_loc[0]
            df.loc[cleft_df.index, 'con_y'] = cluster_loc[1]
            df.loc[cleft_df.index, 'con_x'] = cluster_loc[2]
            df.loc[cleft_df.index, 'area'] = area
            if split_above is not None and len(pre_locs) > 1:
                if area > split_above:
                    # Split large synaptic cleft clusters into two
                    kmeans = KMeans(n_clusters=2)
                    kmeans.fit(pre_locs)
                    y_kmeans = kmeans.predict(pre_locs)
                    middles = kmeans.cluster_centers_
                    middles = [
                        ((middle / voxel_size) * voxel_size).astype(np.int) for
                        middle in middles]
                    for ii, c in enumerate(y_kmeans):
                        df.loc[cleft_df.index[ii], 'con_z'] = middles[c][0]
                        df.loc[cleft_df.index[ii], 'con_y'] = middles[c][1]
                        df.loc[cleft_df.index[ii], 'con_x'] = middles[c][2]

    orphan = df[df.cleft_id == 0]
    if len(orphan) > 0:
        df.loc[orphan.index, 'con_z'] = df.loc[orphan.index, 'pre_z']
        df.loc[orphan.index, 'con_y'] = df.loc[orphan.index, 'pre_y']
        df.loc[orphan.index, 'con_x'] = df.loc[orphan.index, 'pre_x']
        df.loc[orphan.index, 'area'] = 0
    return df


def cluster(synapses_file, cleftfile, cleftds,
            read_roi, voxel_size, roishift=(0, 0, 0),
            cc_thr=130, split_above=None):
    '''Function to cluster synapses based on synaptic cleft.

    This function reads in synapses (pre-loc and post-loc) and clefts (array),
    finds cc in clefts and computes the intersection of clefts and synapses.

    Args:
        synapses_file:
            path to synapses
        cleftfile:
            location of synaptic clefts. All formats that can be read by daisy are possible.
        cleftds:
            datasetname of synaptic clefts
        read_roi:
            daisy.ROI to process
        voxel_size: Voxel size of dataset
        roishift:
            When clefts are shifted relative to synapses, set here the
            corresponding offset. (very specific to n5-to-CATMAID shift, usually
            not needed.
        cc_thr:
            Where to threshold synaptic cleft array to compute connected
            components.
        split_above: If synaptic cleft area is bigger than split_above, compute
            two separate clusters.

    :returns: pandas.Dataframe, only synapses which are fully contained in roi are
        used and returned. pandas.Dataframe has columns:
        pre_x, pre_y, pre_z: pre-location of synapse
        post_x, post_y, post_z: post-location of synapse
        scores: synful score of synapse
        synful_id: original synful id of synapse
        cleft_id: an unique cleft id that the synapse intersected with; if there was no intersection,
            cleft_id is set to zero.
        cleft_score: max value of synaptic cleft array that was found between
            pre and post location of that synapse
        con_x, con_y, con_z:  the computed connector node location. When synapse
            intersected with a synaptic cleft, the closest pre-loc to the
            centroid of the synaptic cleft was used. If there was no intersection,
            connector location is same as pre-location.
        area: number of synaptic cleft cc that synapse intersected with. Set to
            zero for orphan synapses (synapses with no cleft intersection).

    '''
    # Prepare Synapses
    if synapses_file.endswith('.db'):
        conn = sqlite3.connect(sql_synapses_file)
        c = conn.cursor()
        start = time.time()
        links = get_links_in_roi(read_roi, c)
        end = time.time()

    else:
        start = time.time()
        synapses = synapse.read_synapses_in_roi(synapses_file, read_roi)
        links = pd.DataFrame([{'scores': syn.score,
                               'pre_x': syn.location_pre[2],
                               'pre_y': syn.location_pre[1],
                               'pre_z': syn.location_pre[0],
                               'post_x': syn.location_post[2],
                               'post_y': syn.location_post[1],
                               'post_z': syn.location_post[0],
                               'synful_id': syn.id
                               } for syn in synapses])
        end = time.time()

    seconds = end - start
    logger.debug(
        'found {} synapses in roi {}, took {:0.2f} to load'.format(len(links),
                                                                   read_roi,
                                                                   seconds))

    if len(links) == 0:
        logger.info('No synapses found')
        return links

    # Prepare Clefts
    ds = daisy.open_ds(cleftfile, cleftds)
    if ds.voxel_size != voxel_size:
        logger.debug('Resetting voxel size and roi, because it is assumed that '
                     'daisy failed to infer the correct one')
        ds.voxel_size = voxel_size
        ds.roi *= tuple(voxel_size)

    start = time.time()
    roi_cleft = read_roi + roishift
    data = ds[roi_cleft]
    data.materialize()
    data = data.to_ndarray()
    end = time.time()
    logger.debug('Took {:0.2f} to load clefts'.format(end - start))

    start = time.time()
    cc, cleft_id_reg = cc_clefts(data, cc_thr)
    end = time.time()
    logger.debug('Took {:0.2f} to cc'.format(end - start))

    start = time.time()
    links = intersect_clefts_and_synapses(cc, links, read_roi, voxel_size,
                                          statsdata=data)
    end = time.time()
    logger.debug('Took {:0.2f} to intersect'.format(end - start))

    links = add_connector_loc(links, cleft_id_reg, read_roi, voxel_size,
                              split_above=split_above)
    return links
