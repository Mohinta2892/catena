"""
This is a modified predict script that replaces MongoDB and DaisyRequestBlocks usage with Scan. However, portions of the
original script have been commented out instead of being completely removed.
For a leaner version of this, try `predict_scan.py`.
Author: Samia Mohinta
Affiliation: Cambridge University, UK
"""
from __future__ import print_function

import json
import logging
import os
import sys

import gunpowder as gp
import numpy as np
import pymongo

from synful.gunpowder import IntensityScaleShiftClip  # only vectors get clipped
from synful.gunpowder import Train, Predict
from torch_models_loss import *  # torch models
from utils import *
import daisy  # this is daisy 0.3


# for daisy
def block_done_callback(
        db_host,
        db_name,
        # worker_config,
        block,
        start,
        duration):
    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    # worker_config is not passed as of now.
    # TODO find a fix later
    # document = dict(worker_config)
    document = {}
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
        num_workers,
        # worker_config,
        # network_config,
        out_properties={},
        **kwargs):
    # `parameter.json` is expected in the folder as is this python script
    setup_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(setup_dir, 'parameter.json'), 'r') as f:
        net_config = json.load(f)

    # voxels
    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])

    # nm
    voxel_size = gp.Coordinate((40, 4, 4))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')  # this is logits
    pred_post_indicator_sigmoid = gp.ArrayKey(
        'PRED_POST_INDICATOR_SIGMOID')  # this is sigmoid(logits); we want this mask

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_post_indicator, output_size)
    chunk_request.add(pred_post_indicator_sigmoid, output_size)

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
        raise RuntimeError('unknown input data format {}'.format(raw_file))

    # determine total roi of the source raw
    # at this stage pipeline should only consist of the data source as is defined above
    with gp.build(pipeline):
        raw_roi = pipeline.spec[raw].roi

    # # perhaps i don't need context yet???
    context = (input_size - output_size) / 2
    total_output_roi = raw_roi  # .grow(-context, -context)
    print(f"total output roi: {total_output_roi}")
    print(f"total output roi shape: {total_output_roi.get_shape()}")

    # turns RAW SPEC to None
    # pipeline += gp.Pad(raw, size=None)

    pipeline += gp.Normalize(raw)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    # model
    model = STMaskSynfulModel(
        in_channels=1,
        num_fmaps=net_config["fmap_num"],
        fmap_inc_factor=net_config["fmap_inc_factor"],
        downsample_factors=net_config["downsample_factors"],
    )
    # set the model to eval
    model.eval()
    print(model)

    pipeline += Unsqueeze(raw)
    pipeline += Unsqueeze(raw)
    # stack
    # pipeline += Stack(1)

    pipeline += Predict(
        model=model,
        checkpoint=os.path.join(setup_dir, 'model_checkpoint_%d' % iteration),
        inputs={
            'input': raw  # `input` is the name of the argument in the forward of the model
        },
        outputs={
            0: pred_post_indicator,  # logits
            1: pred_post_indicator_sigmoid,
        },
    )

    # This is a recurring theme in all prediction scripts for pretrained networks.
    # Have deleted the intensity scale shifts for the vectors since this script only produces masks
    if m_property is not None and 'scale' in m_property:
        if m_property['scale'] != 1:
            pipeline += gp.IntensityScaleShift(pred_post_indicator,
                                               m_property['scale'], 0)

    # Squeeze the batch x channel dims of any output arrays that are being saved here.
    # That's because it expects a predefined shape as defined below through `prepare ds`
    pipeline += Squeeze(pred_post_indicator_sigmoid)
    pipeline += Squeeze(pred_post_indicator_sigmoid)

    # prepare outputs since otherwise:
    # RuntimeError: Dataset volumes/pred_syn_indicator_mask does not exist in ./sample_C_padded_20160501.zarr,
    # and no ROI is provided for PRED_POST_INDICATOR_SIGMOID. I don't know how to initialize the dataset.
    daisy.prepare_ds(
        out_file,
        "volumes/pred_syn_indicator_mask",
        daisy.Roi(total_output_roi.get_offset(), total_output_roi.get_shape()),
        voxel_size,  # (40,4,4): it's hard-coded above
        np.uint8,
        write_size=output_size
    )

    pipeline += gp.ZarrWrite(
        dataset_names={
            pred_post_indicator_sigmoid: 'volumes/pred_syn_indicator_mask',
        },
        output_filename=out_file,
        dataset_dtypes={
            raw: gp.ArraySpec(roi=total_output_roi)}
    )

    # pipeline += gp.PrintProfilingStats(every=10)

    # Daisy Version: 0.3
    # Commented because ERROR:daisy.context:DAISY_CONTEXT environment variable not found!
    # pipeline += gp.DaisyRequestBlocks(
    #     chunk_request,
    #     roi_map={
    #         raw: 'read_roi',
    #         pred_post_indicator: 'write_roi'
    #     },
    #     num_workers=num_workers,
    #     block_done_callback=lambda b, s, d: block_done_callback(
    #         db_host,
    #         db_name,
    #         # worker_config,
    #         b, s, d))

    # instead using scan to scan through the volumes
    pipeline += gp.Scan(chunk_request)
    predict_request = gp.BatchRequest()

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(
        logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(
        iteration=run_config["iteration"],
        raw_file=run_config["raw_file"],
        raw_dataset=run_config["raw_dataset"],
        out_file=run_config["out_filename"],
        db_host=run_config["db_host"],
        db_name=run_config["db_name"],
        out_properties=run_config["out_properties"],
        num_workers=run_config["num_workers"]
    )
