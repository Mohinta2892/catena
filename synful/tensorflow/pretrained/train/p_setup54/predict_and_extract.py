from __future__ import print_function

import json
import os
import sys
import sys

import gunpowder as gp
import numpy as np
import pymongo
import logging

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

import synful
from synful.gunpowder import ExtractSynapses
from synful.gunpowder import IntensityScaleShiftClip


def block_done_callback(
        db_host,
        db_name,
        worker_config,
        block,
        start,
        duration):
    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    document = dict(worker_config)
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })

    collection.insert_one(document)

    print("Recorded block-done for %s" % (block,))


def predict(
        iteration,
        raw_file,
        raw_dataset,
        db_host,
        db_name,
        out_dir,
        worker_config,
        extraction_parameters,
        networkconfig='train',
        synapse_context=40,
        **kwargs):
    # Load synapse extraction parameters
    with open(extraction_parameters, 'r') as f:
        parameter_dic = json.load(f)

    extractionsettings = synful.detection.SynapseExtractionParameters(
        extract_type=parameter_dic['extract_type'],
        cc_threshold=parameter_dic['cc_threshold'],
        loc_type=parameter_dic['loc_type'],
        score_thr=parameter_dic['score_thr'],
        score_type=parameter_dic['score_type'],
        nms_radius=parameter_dic['nms_radius']
    )

    setup_dir = os.path.dirname(os.path.realpath(__file__))

    with open(
            os.path.join(setup_dir, '{}_net_config.json'.format(networkconfig)),
            'r') as f:
        net_config = json.load(f)

    # voxels
    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])

    # nm
    voxel_size = gp.Coordinate((8, 8, 8))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    if not (type(synapse_context) == list or type(synapse_context) == tuple):
        synapse_context = [synapse_context] * 3
    output_size -= gp.Coordinate(synapse_context) * 2

    parameterfile = os.path.join(setup_dir, 'parameter.json')
    if os.path.exists(parameterfile):
        with open(parameterfile, 'r') as f:
            parameters = json.load(f)
    else:
        parameters = {}

    raw = gp.ArrayKey('RAW')
    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')
    pred_postsyn = gp.PointsKey('PRED_POSTSYN')
    pred_presyn = gp.PointsKey('PRED_PRESYN')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_presyn, output_size)
    chunk_request.add(pred_postsyn, output_size)

    # Hdf5Source
    if raw_file.endswith('.hdf') or raw_file.endswith('.h5'):
        pipeline = gp.Hdf5Source(
            raw_file,
            datasets={
                raw: raw_dataset
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True),
            }
        )
    elif raw_file.endswith('.zarr') or raw_file.endswith('.n5'):
        pipeline = gp.ZarrSource(
            raw_file,
            datasets={
                raw: raw_dataset
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True),
            }
        )
    else:
        raise RuntimeError('unknwon input data format {}'.format(raw_file))

    pipeline += gp.Pad(raw, size=None)

    pipeline += gp.Normalize(raw)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.tensorflow.Predict(
        os.path.join(setup_dir, 'train_net_checkpoint_{}'.format(iteration)),
        inputs={
            net_config['raw']: raw,
        },
        outputs={
            net_config['pred_syn_indicator_out']: pred_post_indicator,
            net_config['pred_partner_vectors']: pred_postpre_vectors
        },
        graph=os.path.join(setup_dir, '{}_net.meta'.format(networkconfig))
    )

    d_scale = parameters['d_scale'] if 'd_scale' in parameters else None
    if d_scale != 1 and d_scale is not None:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           1. / d_scale,
                                           0)  # Map back to nm world.

    pipeline += ExtractSynapses(pred_post_indicator,
                                pred_postpre_vectors,
                                pred_postsyn, pred_presyn,
                                settings=extractionsettings,
                                context=synapse_context,
                                out_dir=out_dir)

    pipeline += gp.PrintProfilingStats()

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: 'read_roi',
            pred_presyn: 'write_roi',
            pred_postsyn: 'write_roi'
        },
        num_workers=worker_config['num_cache_workers'],
        block_done_callback=lambda b, s, d: block_done_callback(
            db_host,
            db_name,
            worker_config,
            b, s, d))

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(**run_config)
