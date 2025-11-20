from __future__ import division

from .. import detection, synapse, database
from funlib.math import cantor_number
from gunpowder import BatchFilter
from gunpowder.contrib.points import PreSynPoint, PostSynPoint
from pymongo import MongoClient
import gunpowder as gp
import logging
import numpy as np
import os
import time

logger = logging.getLogger(__name__)


class ExtractSynapses(BatchFilter):
    '''Extract synaptic partners from 2 prediction channels. One prediction map
    indicates the location (m_array), the second map (d_array) indicates the
    direction to its synaptic partner. Optionally, writes it to a database.

    Args:

        m_array (:class:``ArrayKey``):
            The key of the array to extract points from.

        d_array (:class:``ArrayKey``):
            The key of the array to extract vectors from.

        srcpoints (:class:``PointsKey``):
            The key of the presynaptic points to create. Note, that only those
            synaptic partners will be in roi, where both pre and postsynaptic
            site are fully contained in ROI.

        trgpoints (:class:``PointsKey``):
            The key of the postsynaptic points to create.

        out_dir (``string``):
            The directory to store the extracted synapses in.

        settings (:class:``SynapseExtractionParameters``):
            Which settings to use to extract synapses.

        context (``list`` of ``int`` or ``int``):
            List which defines padding of srcpoints ROI in world units. Synapses
            are detected in this padded ROI, while only written out in original
            srcpoints ROI.

        db_name (``string``):
        db_host (``string``):
            Database name and host to store block statistics.

        pre_to_post (``bool``):
            If set to True, it is assumed that m_array is indicating the
            presence of presynaptic location, and d_array is encoding the
            direction to the postsynaptic partner. If set to False, m_array
            indicates postsynaptic location, and d_array the direction to the
            presynaptic partner. (Only relevant for writing synapses out
            to a database.)
    '''

    def __init__(
            self,
            m_array,
            d_array,
            srcpoints,
            trgpoints,
            out_dir,
            settings=None,
            context=120,
            db_name=None,
            db_host=None,
            pre_to_post=False):
        if db_name is not None or db_host is not None:
            if db_host is None or db_name is None:
                logger.warning(
                    'If synapses are supposed to be written out to database, '
                    'both db_name and db_host must be provided')

        self.m_array = m_array
        self.d_array = d_array
        self.srcpoints = srcpoints
        self.trgpoints = trgpoints
        self.out_dir = out_dir
        self.settings = settings
        if type(context) == tuple:
            context = list(context)
        if not type(context) == list:
            context = [context]
        self.context = context
        self.db_name = db_name
        self.db_host = db_host
        self.pre_to_post = pre_to_post

    def setup(self):

        self.spec_src = gp.PointsSpec()
        self.spec_trg = gp.PointsSpec()

        self.provides(self.srcpoints, self.spec_src)
        self.provides(self.trgpoints, self.spec_trg)

        self.enable_autoskip()

    def prepare(self, request):

        context = self.context
        dims = request[self.srcpoints].roi.dims()

        assert type(context) == list
        if len(context) == 1:
            context = context * dims

        # request array in a larger area to get predictions from outside
        # write roi
        m_roi = request[self.srcpoints].roi.grow(
            gp.Coordinate(context),
            gp.Coordinate(context))

        # however, restrict the request to the array actually provided
        # m_roi = m_roi.intersect(self.spec[self.m_array].roi)
        request[self.m_array] = gp.ArraySpec(roi=m_roi)

        # Do the same for the direction vector array.
        request[self.d_array] = gp.ArraySpec(roi=m_roi)

    def process(self, batch, request):

        srcpoints, trgpoints = self.__extract_synapses(batch, request)

        points_spec = self.spec[self.srcpoints].copy()
        points_spec.roi = request[self.srcpoints].roi
        batch.points[self.srcpoints] = gp.Points(data=srcpoints,
                                                 spec=points_spec)
        batch.points[self.trgpoints] = gp.Points(data=trgpoints,
                                                 spec=points_spec.copy())

        # restore requested arrays
        if self.m_array in request:
            batch.arrays[self.m_array] = batch.arrays[self.m_array].crop(
                request[self.m_array].roi)
        if self.d_array in request:
            batch.arrays[self.d_array] = batch.arrays[self.d_array].crop(
                request[self.d_array].roi)

    def __extract_synapses(self, batch, request):
        mchannel = batch[self.m_array]
        dchannel = batch[self.d_array]
        start_time = time.time()

        if self.db_name is not None and self.db_host is not None:
            srcroi = request[self.srcpoints].roi
            begin = srcroi.get_begin()
            batch_id = cantor_number(begin/mchannel.spec.voxel_size)
            cl = MongoClient(self.db_host)
            b_status = cl[self.db_name]['blocks_status']
            res = b_status.find_one({'batch_id': batch_id})
            if res:
                overwrite = True
                res.update({'status': 2})  # 2 --> started, 1 --> complete
                logging.debug('overwriting partially written block with '
                             'batch id {}'.format(batch_id))
            else:
                overwrite = False
                res = {
                    'batch_id': batch_id,
                    'status': 2
                }
                logging.debug('setting batch {} status to 2'.format(batch_id))
            b_status.replace_one({'batch_id': batch_id}, res, upsert=True)

        predicted_syns, scores = detection.find_locations(mchannel.data,
                                                          self.settings,
                                                          mchannel.spec.voxel_size)
        logger.debug('find locations %0.2f' % (time.time() - start_time))
        # Filter synapses for scores.
        new_scorelist = []
        if self.settings.score_thr is not None:
            filtered_list = []
            for ii, loc in enumerate(predicted_syns):
                score = scores[ii]
                if score > self.settings.score_thr:
                    filtered_list.append(loc)
                    new_scorelist.append(score)

            logger.debug(
                'filtered out %i' % (len(predicted_syns) - len(filtered_list)))
            predicted_syns = filtered_list
            scores = new_scorelist
        start_time = time.time()
        target_sites = detection.find_targets(predicted_syns, dchannel.data,
                                              voxel_size=dchannel.spec.voxel_size)
        logger.debug('find targets %0.2f' % (time.time() - start_time))

        # Synapses need to be shifted to the global ROI
        # (currently aligned with arrayroi)
        for loc in predicted_syns:
            loc += np.array(mchannel.spec.roi.get_begin())
        for loc in target_sites:
            loc += np.array(dchannel.spec.roi.get_begin())

        if self.pre_to_post:
            synapses = synapse.create_synapses(predicted_syns, target_sites,
                                               scores=scores)
        else:
            synapses = synapse.create_synapses(target_sites, predicted_syns,
                                               scores=scores)

        srcroi = request[self.srcpoints].roi

        ids, positions, scores = self.__create_node_arrays(
            synapses,
            voxel_size=dchannel.spec.voxel_size,
            roi=srcroi)

        self.__store_node_arrays(
            self.out_dir,
            ids,
            positions,
            scores,
            srcroi)

        # Bring into gunpowder format
        srcpoints = {}
        trgpoints = {}
        # syn_id = 0
        # for syn in synapses:
            # loc = gp.Coordinate(syn.location_pre)
            # if srcroi.contains(syn.location_pre) and srcroi.contains(
                    # syn.location_post):  # TODO: currently, gunpowder complains
                # # about points being outside ROI, thus can only provide synapses
                # # where pre and point are inside ROI
                # loc_index = syn_id * 2
                # syn_point = PreSynPoint(location=loc,
                                        # location_id=loc_index,
                                        # synapse_id=syn_id,
                                        # partner_ids=[loc_index + 1],
                                        # props={'score': syn.score})
                # srcpoints[loc_index] = syn_point
                # loc = gp.Coordinate(syn.location_post)
                # syn_point = PostSynPoint(location=loc,
                                         # location_id=loc_index + 1,
                                         # synapse_id=syn_id,
                                         # partner_ids=[loc_index],
                                         # props={'score': syn.score})
                # trgpoints[loc_index + 1] = syn_point
                # syn_id += 1
        return srcpoints, trgpoints

    def __create_node_arrays(self, synapses, voxel_size, roi=None):

        # filter synapses
        if roi is not None:
            synapses = [
                synapse
                for synapse in synapses
                if roi.contains(synapse.location_post)
            ]

        num_synapses = len(synapses)

        ids = np.zeros((num_synapses,), dtype=np.uint64)
        positions = np.zeros((num_synapses, 2, 3), dtype=np.int32)
        scores = np.zeros((num_synapses,), dtype=np.float32)

        for i, synapse in enumerate(synapses):

            synapse_id = cantor_number(synapse.location_post/voxel_size)

            ids[i] = synapse_id
            positions[i, 0] = synapse.location_pre
            positions[i, 1] = synapse.location_post
            scores[i] = synapse.score

        return ids, positions, scores

    def __store_node_arrays(self, out_dir, ids, positions, scores, block_roi):

        block_offset = block_roi.get_offset()
        block_dir = os.path.join(
            out_dir,
            str(block_offset[0]),
            str(block_offset[1]))
        block_path = os.path.join(
            block_dir,
            str(block_offset[2]) + '.npz')

        if not os.path.exists(block_dir):
            os.makedirs(block_dir, exist_ok=True)

        np.savez(
            block_path,
            ids=ids,
            positions=positions,
            scores=scores)
