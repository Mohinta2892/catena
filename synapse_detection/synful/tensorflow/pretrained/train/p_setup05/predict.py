from __future__ import print_function

import json
import logging
import os
import sys

import gunpowder as gp
import numpy as np
import pymongo

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
        out_file,
        db_host,
        db_name,
        worker_config,
        network_config,
        out_properties={},
        **kwargs):
    setup_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(setup_dir,
                           '{}_net_config.json'.format(network_config)),
              'r') as f:
        net_config = json.load(f)

    # voxels
    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])

    # nm
    voxel_size = gp.Coordinate((40, 4, 4))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    parameterfile = os.path.join(setup_dir, 'parameter.json')
    if os.path.exists(parameterfile):
        with open(parameterfile, 'r') as f:
            parameters = json.load(f)
    else:
        parameters = {}

    raw = gp.ArrayKey('RAW')
    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_postpre_vectors, output_size)

    d_property = out_properties[
        'pred_partner_vectors'] if 'pred_partner_vectors' in out_properties else None


    # Hdf5Source
    if raw_file.endswith('.hdf'):
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
        os.path.join(setup_dir, 'train_net_checkpoint_%d' % iteration),
        inputs={
            net_config['raw']: raw
        },
        outputs={
            net_config['pred_partner_vectors']: pred_postpre_vectors
        },
        graph=os.path.join(setup_dir, '{}_net.meta'.format(network_config))
    )
    d_scale = parameters['d_scale'] if 'd_scale' in parameters else None
    if d_scale != 1 and d_scale is not None:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           1. / d_scale,
                                           0)  # Map back to nm world.

    if d_property is not None and 'scale' in d_property:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           d_property['scale'], 0)
    if d_property is not None and 'dtype' in d_property:
        assert d_property['dtype'] == 'int8' or d_property[
            'dtype'] == 'float32', 'predict not adapted to dtype {}'.format(
            d_property['dtype'])
        if d_property['dtype'] == 'int8':
            pipeline += IntensityScaleShiftClip(pred_postpre_vectors,
                                                1, 0, clip=(-128, 127))

    pipeline += gp.ZarrWrite(
        dataset_names={
            pred_postpre_vectors: 'volumes/pred_partner_vectors',
        },
        output_filename=out_file
    )

    pipeline += gp.PrintProfilingStats(every=10)

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: 'read_roi',
            pred_postpre_vectors: 'write_roi',
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
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(
        logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(**run_config)
