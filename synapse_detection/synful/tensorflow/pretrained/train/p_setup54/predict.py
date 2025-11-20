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

    raw = gp.ArrayKey('RAW')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_post_indicator, output_size)

    m_property = out_properties[
        'pred_syn_indicator_out'] if 'pred_syn_indicator_out' in out_properties else None

    # Select Source based on filesuffix.
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
    pipeline += gp.ZarrWrite(
        dataset_names={
            raw: 'volumes/raw',
        },
        output_filename=out_file
    )
    pipeline += gp.Normalize(raw)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.tensorflow.Predict(
        os.path.join(setup_dir, 'train_net_checkpoint_%d' % iteration),
        inputs={
            net_config['raw']: raw
        },
        outputs={
            net_config['pred_syn_indicator_out']: pred_post_indicator,
        },
        graph=os.path.join(setup_dir, '{}_net.meta'.format(network_config))
    )

    if m_property is not None and 'scale' in m_property:
        if m_property['scale'] != 1:
            pipeline += gp.IntensityScaleShift(pred_post_indicator,
                                               m_property['scale'], 0)

    pipeline += gp.ZarrWrite(
        dataset_names={
            pred_post_indicator: 'volumes/pred_syn_indicator',
        },
        output_filename=out_file
    )

    pipeline += gp.PrintProfilingStats(every=10)

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: 'read_roi',
            pred_post_indicator: 'write_roi'
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
