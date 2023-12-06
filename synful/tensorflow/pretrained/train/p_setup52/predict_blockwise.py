from funlib.run import run
import hashlib
import json
import logging
import os
import sys
import time
import glob

import daisy
import numpy as np
import pymongo

logging.basicConfig(level=logging.INFO)


def predict_blockwise(
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        out_directory,
        out_filename,
        num_workers,
        db_host,
        db_name,
        worker_config=None,
        out_properties={},
        overwrite=False,
        configname='test'):
    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):

            Input raw file for network to predict from.

        raw_dataset (``string``):

            Datasetname of raw data.

        out_directory (``string``):

            Output base directory.

        out_filename (``string``):

            File is written to <out_directory>/<setup>/<iteration>/<out_filename>

        num_workers (``int``):

            How many blocks to run in parallel.

        db_host (``string``):

            MongoDB host. This is used to monitor block completeness.

        db_name (``db_name``):

            MongoDB name. A collection is created with `blocks_predicted` for
            monitoring blocks.

        out_properties (``dic``, optional):

            Use this to set properties for the output data. The dictionary
            maps from tensorflow name to properties (all optional):
            - dsname: sets the datasetname, if not provided, tensorflow name is used
            - dtype: sets the dtype (make sure that you scale and clip
            accordingly). If not provided, the dtype in config file is used.
            - scale: scales the data and sets dataset attribute 'scale'

        configname (``string``, optional):

            Name of the configfile: Networksetups (such as input_size and
            output_size) are loaded from file: <configname>_net_config.json.
            This should be the same file that the predict script loads.
            Train usually indicates a smaller network than test (test network
            is written out for larger datasets/production).

        overwrite (``bool``, optional):

            If set to True, inference is started form scratch and log info
            from `blocks_predicted` is ignored, database collection
            `blocks_predicted` is overwritten.

    '''

    experiment_dir = '../'
    # train_dir = os.path.join(experiment_dir, 'train', experiment)
    # if not os.path.exists(train_dir):
    # train_dir = os.path.join(experiment_dir, 'train')
    train_dir = '../'
    db_name = db_name + '_{}_{}'.format(setup, iteration)
    if experiment != 'cremi':
        db_name += f'_{experiment}'

    network_dir = os.path.join(experiment, setup, str(iteration))
    if experiment != 'cremi':  # backwards compatability
        out_directory = os.path.join(out_directory, experiment)

    raw_file = os.path.abspath(raw_file)
    out_file = os.path.abspath(
        os.path.join(out_directory, setup, str(iteration), out_filename))

    setup = os.path.abspath(os.path.join(train_dir, setup))

    print('Input file path: ', raw_file)
    print('Output file path: ', out_file)
    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    try:
        source = daisy.open_ds(raw_file, raw_dataset)
    except:
        raw_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, raw_dataset)
    print("Source dataset has shape %s, ROI %s, voxel size %s" % (
        source.shape, source.roi, source.voxel_size))

    # load config
    with open(
            os.path.join(setup, '{}_net_config.json'.format(configname))) as f:
        print("Reading setup config from %s" % os.path.join(setup,
                                                            '{}_net_config.json'.format(
                                                                configname)))
        net_config = json.load(f)
    outputs = net_config['outputs']

    # get chunk size and context
    net_input_size = daisy.Coordinate(
        net_config['input_shape']) * source.voxel_size
    net_output_size = daisy.Coordinate(
        net_config['output_shape']) * source.voxel_size
    context = (net_input_size - net_output_size) / 2

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    print("Following sizes in world units:")
    print("net input size  = %s" % (net_input_size,))
    print("net output size = %s" % (net_output_size,))
    print("context         = %s" % (context,))

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    print("Following ROIs in world units:")
    print("Block read  ROI  = %s" % block_read_roi)
    print("Block write ROI  = %s" % block_write_roi)
    print("Total input  ROI  = %s" % input_roi)
    print("Total output ROI  = %s" % output_roi)

    # Adding the raw to the output too, to prevent loading separately!
    # Yes data duplication, but convenient!
    # Initially FileNotFoundErrors can occur for reasons still unclear to us, but it should still run and save the raw.
    ds = daisy.prepare_ds(
            out_file,
            "volumes/raw",
            output_roi,
            source.voxel_size,
            np.uint8,
            write_roi=block_write_roi,
            # temporary fix until
            # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
            # (we want gzip to be the default)
            compressor={'id': 'gzip', 'level': 5}
        )
	
    logging.info('Preparing output dataset')
    print("Preparing output dataset...")
    for outputname, val in outputs.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        scale = None
        print(outputname)
        if outputname in out_properties:
            out_property = out_properties[outputname]
            out_dtype = out_property[
                'dtype'] if 'dtype' in out_property else out_dtype
            scale = out_property['scale'] if 'scale' in out_property else None
            outputname = out_property[
                'dsname'] if 'dsname' in out_property else outputname
        print('setting dtype to {}'.format(out_dtype))
        out_dataset = 'volumes/%s' % outputname
        print('Creatining dataset: {}'.format(out_dataset))
        print('Number of dimensions is %i' % out_dims)
        ds = daisy.prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            # temporary fix until
            # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
            # (we want gzip to be the default)
            compressor={'id': 'gzip', 'level': 5}
        )
        if scale is not None:
            ds.data.attrs['scale'] = scale

    print("Starting block-wise processing...")

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    if overwrite:
        db.drop_collection('blocks_predicted')

    if 'blocks_predicted' not in db.list_collection_names():
        blocks_predicted = db['blocks_predicted']
        blocks_predicted.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')
    else:
        blocks_predicted = db['blocks_predicted']

    # process block-wise
    succeeded = daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(
            setup,
            network_dir,
            iteration,
            raw_file,
            raw_dataset,
            out_file,
            out_properties,
            db_host,
            db_name,
            configname,
            worker_config),
        check_function=lambda b: check_block(
            blocks_predicted,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")


def predict_worker(
        setup,
        network_dir,
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_properties,
        db_host,
        db_name,
        network_config,
        worker_config):
    setup_dir = os.path.join('..', 'train', setup)
    predict_script = os.path.abspath(
        os.path.join(setup_dir, 'predict.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']
    if worker_config is None:
        worker_config = {
            'queue': 'gpu_rtx',
            'num_cpus': 2,
            'num_cache_workers': 4,
        }
    else:
        with open(worker_config, 'r') as f:
            worker_config = json.load(f)

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'out_file': out_file,
        'out_properties': out_properties,
        'db_host': db_host,
        'db_name': db_name,
        'worker_config': worker_config,
        'network_config': network_config
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s' % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.predict_blockwise', network_dir)

    try:
        os.makedirs(output_dir)
    except:
        pass

    config_file = os.path.join(output_dir, '%d.config' % config_hash)

    log_out = os.path.join(output_dir, 'predict_blockwise_%d.out' % worker_id)
    log_err = os.path.join(output_dir, 'predict_blockwise_%d.err' % worker_id)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    print("Running block with config %s..." % config_file)

    # Commented out: Use this if you have a cluster available using lsf (--> command bsub).
    # command = run(
    #     command='python -u %s %s' % (
    #         predict_script,
    #         config_file),
    #     queue=worker_config['queue'],
    #     num_cpus=worker_config['num_cpus'],
    #     num_gpus=1,
    #     mount_dirs=[
    #     ],
    #     execute=False,
    #     expand=False)
    # daisy.call(command, log_out=log_out, log_err=log_err)
    command = "python -u %s %s" % (predict_script, config_file)
    os.system(command)
    logging.info('Predict worker finished')


def check_block(blocks_predicted, block):
    # done = blocks_predicted.count({'block_id': block.block_id}) >= 1
    done = len(list(blocks_predicted.find({"block_id": block.block_id}))) >= 1
    return done


if __name__ == "__main__":

    config_file = sys.argv[1]

    if config_file.endswith('.json'):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print('loaded config file')
        if 'overwrite' in config:
            config['overwrite'] = bool(config['overwrite'])
        start = time.time()

        predict_blockwise(**config)

        end = time.time()

        seconds = end - start
        print('Total time to predict: %f seconds' % seconds)
    else:
        print('assuming directory, reading config files from directory')
        configs = glob.glob(config_file + '/*.json')
        print('processing {} config files'.format(len(configs)))
        for ii, config_file in enumerate(configs):
            print('processing {}: {}/{}'.format(config_file, ii, len(configs)))
            with open(config_file, 'r') as f:
                config = json.load(f)
            print('loaded config file')
            if 'overwrite' in config:
                config['overwrite'] = bool(config['overwrite'])
            start = time.time()

            predict_blockwise(**config)

            end = time.time()

            seconds = end - start
