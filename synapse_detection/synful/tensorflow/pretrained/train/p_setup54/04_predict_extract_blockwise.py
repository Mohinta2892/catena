from funlib.run import run
from synful import database, detection
import daisy
import glob
import hashlib
import json
import numpy as np
import os
import pymongo
import sys
import time
import logging

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logging.basicConfig(level=logging.INFO)


# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)


def predict_blockwise(
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        num_workers,
        db_host,
        db_name,
        out_basedir,
        extraction_parameters,
        overwrite=False,
        configname='test',
        synapse_context=40,
        mask=None,
        mask_ds=None,
        worker_config=None,
        max_retries=2
):
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

        num_workers (``int``):

            How many blocks to run in parallel.

        db_host (``string``):

            MongoDB host. This is used to monitor block completeness.

        db_name (``db_name``):

            MongoDB name. A collection is created with `blocks_predicted` for
            monitoring blocks.

        out_basedir (``string``):

            Directory name to which synapses are written.

        extraction_parameters (``string``):

                Path to json file that sets the synapse extraction parameters.

        overwrite (``bool``, optional):

            If set to True, inference is started form scratch and log info
            from `blocks_predicted` is ignored, database collection
            `blocks_predicted` is overwritten.


        configname (``string``, optional):

            Name of the configfile: Networksetups (such as input_size and
            output_size) are loaded from file: <configname>_net_config.json.
            This should be the same file that the predict script loads.
            Train usually indicates a smaller network than test (test network
            is written out for larger datasets/production).

        synapse_context (``list`` of ``int``, optional):

            Defines padding of synapse write ROI. Important to avoid border
            effects when extracting synapses.

        mask (``string``):

            Mask filename. When provided, blocks are skipped where mask==0.

        mask_ds (``string``):

            Datasetname of mask.

        worker_config (``string``):

            Name of workerconfigfile.

        max_retries (``int``) :

            How often daisy should retry to process a block after failure.

    '''

    if overwrite:
        print('writing to base dir {} using setup {}'.format(out_basedir, setup))
        print(
            "WARNING: you are about to overwrite data of previous runs. "
            "Press ENTER if you are sure you want to continue.")
        input()

    experiment_dir = '../'
    # train_dir = os.path.join(experiment_dir, '02_train', experiment)
    # if not os.path.exists(train_dir):
    # train_dir = os.path.join(experiment_dir, '02_train')
    train_dir = '/home/pretrained/train/code'
    db_name = db_name + '_{}_{}'.format(setup.replace('/', '_'), iteration)

    if db_host.endswith('.json'):
        logging.info('db_host is loaded from json file {}'.format(db_host))
        with open(db_host) as f:
            db_host = json.load(f)['db_host']

    network_dir = os.path.join(experiment, setup, str(iteration))

    raw_file = os.path.abspath(raw_file)

    setup = os.path.abspath(os.path.join(train_dir, setup))

    print('Input file path: ', raw_file)
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

    if not (type(synapse_context) == list or type(synapse_context) == tuple):
        synapse_context = [synapse_context] * 3

    net_output_size = net_output_size - daisy.Coordinate(
        synapse_context) * 2
    context = (net_input_size - net_output_size) / 2

    # get total input and output ROIs
    adjusted_roi = source.roi.snap_to_grid(net_output_size)
    input_roi = adjusted_roi.grow(context, context)
    output_roi = adjusted_roi

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

    print("Starting block-wise processing...")

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    if overwrite:
        db.drop_collection('blocks_predicted')
        db.drop_collection('blocks_masked')
        db.drop_collection('blocks_status')

    if 'blocks_predicted' not in db.list_collection_names():
        blocks_predicted = db['blocks_predicted']
        blocks_predicted.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')
        blocks_masked = db['blocks_masked']
        blocks_masked.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')
        blocks_status = db['blocks_status']
        blocks_status.create_index(
            [('batch_id', pymongo.ASCENDING)],
            name='batch_id')

    else:
        blocks_predicted = db['blocks_predicted']
        blocks_masked = db['blocks_masked']
        blocks_status = db['blocks_status']

    with open(extraction_parameters, 'r') as f:
        parameter_dic = json.load(f)
    parameters = detection.SynapseExtractionParameters(
        extract_type=parameter_dic['extract_type'],
        cc_threshold=parameter_dic['cc_threshold'],
        loc_type=parameter_dic['loc_type'],
        score_thr=parameter_dic['score_thr'],
        score_type=parameter_dic['score_type'],
        nms_radius=parameter_dic['nms_radius']
    )

    out_dir = 'syn'
    if parameters.extract_type == 'cc':
        out_dir = out_dir + '_{}_thr{:06d}_{}'.format(
            parameters.extract_type,
            int(parameters.cc_threshold * 100000),
            parameters.score_type)
    else:
        out_dir = out_dir + '_{}_rad{}_{}_{}'.format(
            parameters.extract_type,
            parameters.nms_radius[0],
            parameters.nms_radius[1],
            parameters.nms_radius[2])

    if overwrite:
        print("Overwriting: {}:{}:{}".format(db_host, db_name, out_dir))
        dag_db = database.DAGDatabase(db_name, db_host, db_col_name=out_dir,
                                      mode='w')
    else:
        dag_db = database.DAGDatabase(db_name, db_host, db_col_name=out_dir,
                                      mode='r')

    # Write parameters
    dag_db.collection['parameters'].insert_one(
        parameter_dic
    )

    out_dir = os.path.join(out_basedir, db_name, out_dir)
    print("Writing synpases to {}".format(out_dir))

    # Prepare npz-directory.
    if overwrite and os.path.exists(out_dir):
        print(
            f"Please manually remove {out_dir}. Exiting")
        quit()

    attrfile = os.path.join(out_dir, 'synfulattrs.json')
    if os.path.exists(attrfile):
        with open(attrfile) as f:
            attrdic = json.load(f)
            chunk_size = attrdic.get("chunk_size", None)
            if chunk_size is not None:
                if not (tuple(chunk_size) == net_output_size):
                    print(
                        f"Please manually remove {out_dir}. Datasets are not compatible: Chunk size is different.")
                    exit()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(attrfile, 'w') as f:
        json.dump({
            "chunk_size": net_output_size
        }, f)

    if mask == "":  # changed here to handle None
        mask = None
        mask_ds = None

    if mask is not None:  # changed here to handle None
        assert mask_ds is not None, 'mask_ds necassary if mask is set'
        mask = daisy.open_ds(mask, mask_ds)

    processed_block_ids = get_processed_block_ids(
        blocks_predicted,
        blocks_masked)

    # process block-wise
    succeeded = daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(
            experiment,
            setup,
            network_dir,
            iteration,
            raw_file,
            raw_dataset,
            db_host,
            db_name,
            out_dir,
            extraction_parameters,
            configname,
            synapse_context,
            worker_config=worker_config),
        check_function=lambda b: check_block(
            processed_block_ids,
            blocks_predicted,
            b,
            mask=mask,
            blocks_masked=blocks_masked),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang',
        max_retries=max_retries)

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")


def get_processed_block_ids(blocks_predicted, blocks_masked):
    logging.info("Getting IDs of already processed blocks...")
    start = time.time()

    predicted = blocks_predicted.find(projection={'block_id': True})
    predicted_ids = set([b['block_id'] for b in predicted])

    if blocks_masked is not None:
        masked = blocks_masked.find(projection={'block_id': True})
        masked_ids = set([b['block_id'] for b in masked])
        predicted_ids.update(masked_ids)

    logging.info("Got processed block IDs in %.3fs", (time.time() - start))

    return predicted_ids


def predict_worker(
        experiment,
        setup,
        network_dir,
        iteration,
        raw_file,
        raw_dataset,
        db_host,
        db_name,
        out_dir,
        extraction_parameters,
        configname,
        synapse_context,
        worker_config=None):
    setup_dir = os.path.join('..', '02_train', setup)
    if experiment != 'cremi':  # Backwards compability.
        setup_dir = os.path.join('..', '02_train', experiment, setup)

    predict_script = os.path.abspath(
        os.path.join(setup_dir, 'predict_and_extract.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']
    if worker_config is None:
        worker_config = {
            'queue': 'slowpoke',
            'num_cpus': 2,
            'num_cache_workers': 10,
            'singularity': 'synful/synful_py3:v0.7'
        }
    else:
        with open(worker_config, 'r') as f:
            worker_config = json.load(f)

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'db_host': db_host,
        'db_name': db_name,
        'out_dir': out_dir,
        'worker_config': worker_config,
        'extraction_parameters': extraction_parameters,
        'networkconfig': configname,
        'synapse_context': synapse_context,
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s' % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.predictextract_blockwise', network_dir)

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

    singularity_image = None
    if 'singularity' in worker_config:
        singularity_image = worker_config['singularity']

    # Commented out this here because not using singularity!
    # command = run(
    #     command='python -u %s %s' % (
    #         predict_script,
    #         config_file),
    #     queue=worker_config['queue'],
    #     num_cpus=worker_config['num_cpus'],
    #     num_gpus=1,
    #     singularity_image=singularity_image,
    #     mount_dirs=[
    #         '/nrs',
    #         '/groups'
    #     ],
    #     execute=False,
    #     expand=False)

    command = "python -u %s %s" % (predict_script, config_file)

    # daisy.call(command, log_out=log_out, log_err=log_err)
    os.system(command)

    logging.info('Predict worker finished')

    # # if things went well, remove temporary files
    # os.remove(config_file)
    # os.remove(log_out)
    # os.remove(log_err)


def check_block(
        processed_block_ids,
        blocks_predicted,
        block,
        blocks_masked=None,
        mask=None):
    logging.debug("Checking if block %s is complete..." % block.write_roi)

    if block.block_id in processed_block_ids:
        return True
    # changed because count() is no longer supported
    # if blocks_predicted.count({'block_id': block.block_id}) >= 1:
    #     return True
    if len(list(blocks_predicted.find({"block_id": block.block_id}))) >= 1:
        return True

    if mask == "":
        mask = None
        mask_ds = None

    if mask is not None:

        # if blocks_masked.count({'block_id': block.block_id}) >= 1:
        if len(list(blocks_predicted.find({"block_id": block.block_id}))) >= 1:
            return True

        write_roi_sn = block.write_roi.snap_to_grid(mask.voxel_size)
        mask_block = mask.to_ndarray(roi=write_roi_sn, fill_value=0)
        if np.max(mask_block) == 0:  # max --> conservative
            blocks_masked.insert_one({
                'block_id': block.block_id
            })
            return True

    return False


if __name__ == "__main__":
    config_file = sys.argv[1]

    if config_file.endswith('.json'):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print('loaded config file')
        if 'overwrite' in config:
            config['overwrite'] = bool(config['overwrite'])
        start = time.time()
        print('using setup {}'.format(config['setup']))
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
