import daisy
import json
import logging
import sys
import time
import os
import numpy as np
import multiprocessing as mp
from funlib.segment.graphs.impl import connected_components
from funlib.persistence import graphs, open_ds
from funlib.geometry import Roi, Coordinate
from yacs.config import CfgNode as CN

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.shared_graph_provider').setLevel(logging.DEBUG)


def find_segments(
        db_host,
        db_name,
        fragments_file,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        fragments_ds=None,
        roi_offset=None,
        roi_shape=None,
        **kwargs):
    """

    Args:

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        fragments_file (``string``):

            Path to the file containing the fragments.

        edges_collection (``string``):

            The name of the MongoDB database collection to use.

        thresholds_minmax (``list`` of ``int``):

            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.

        thresholds_step (``float``):

            The step size to use when generating thresholds between min/max.

        roi_offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    """

    print("Reading graph from DB ", db_name, edges_collection)
    start = time.time()

    graph_provider = graphs.MongoDbGraphProvider(
        db_name,
        db_host,
        nodes_collection=f"{sample_name}_nodes",
        edges_collection=edges_collection,
        position_attribute=[
            'center_z',
            'center_y',
            'center_x'])

    if fragments_ds:
        fragments = open_ds(fragments_file, fragments_ds)

        roi = fragments.roi

    else:

        roi = Roi(
            roi_offset,
            roi_shape)

    print("Getting graph for roi %s" % roi)

    graph = graph_provider.get_graph(roi)

    print("Read graph in %.3fs" % (time.time() - start))

    if graph.number_of_nodes == 0:
        print("No nodes found in roi %s" % roi)
        return

    nodes = np.array(graph.nodes)
    edges = np.stack(list(graph.edges), axis=0)

    scores = np.array([graph.edges[tuple(e)]["merge_score"] for e in edges]).astype(
        np.float32
    )  # adapted from WPatton's mutex_agglomerate `find_segments.py`

    print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    out_dir = os.path.join(os.path.dirname(fragments_file), "luts_full")
    logging.debug(f"Saving luts here {out_dir}")

    # make the lut dir, does not exist
    os.makedirs(out_dir, exist_ok=True)

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    start = time.time()

    for threshold in thresholds:
        get_connected_components(
            nodes,
            edges,
            scores,
            threshold,
            edges_collection,
            out_dir)

        print("Created and stored lookup tables in %.3fs" % (time.time() - start))


def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        edges_collection,
        out_dir,
        **kwargs):
    print("Getting CCs for threshold %.3f..." % threshold)
    start = time.time()
    components = connected_components(nodes, edges, scores, threshold)
    print("%.3fs" % (time.time() - start))

    print("Creating fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()
    lut = np.array([nodes, components])

    print("%.3fs" % (time.time() - start))

    print("Storing fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()

    lookup = 'seg_%s_%d' % (edges_collection, int(threshold * 100))

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)

    print("%.3fs" % (time.time() - start))


if __name__ == "__main__":
    config_file = sys.argv[1]  # take one of the worker config files from daisy_logs??

    # parse the args file to become cfg
    cfg = CN()
    # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)

    start = time.time()

    sample_name = cfg.DATA.SAMPLE_NAME
    db_host = cfg.DATA.DB_HOST
    db_name = cfg.DATA.DB_NAME
    merge_function = 'hist_quant_60'  # cfg.INS_SEGMENT.MERGE_FUNCTION - hardcoded for testing

    find_segments(
        db_host,
        db_name,
        fragments_file=cfg.DATA.SAMPLE,
        edges_collection=sample_name + "_edges_" + merge_function,
        thresholds_minmax=[0.1, 0.8],  # hardcoded: don't know what this means??
        thresholds_step=0.1,
        fragments_ds=cfg.INS_SEGMENT.OUT_FRAGS_DS,
        roi_offset=None,
        roi_shape=None, )

    print('Took %.3f seconds to find segments and store LUTs' % (time.time() - start))
