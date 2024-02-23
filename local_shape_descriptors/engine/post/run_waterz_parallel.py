import json
import hashlib
import logging
import lsd
import numpy as np
import os
import daisy
import sys
import time
import pymongo
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)


def main():
    """ This should call parallel fragments and agglomerate from lsds. Must supply:

        Args:

        affs (`class:daisy.Array`):

            An array containing affinities.

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to write nodes for extracted fragments to. This does
            not yet add adjacency edges, for that, an agglomeration method
            should be called after this function.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction, in world units.

        fragments_out (`class:daisy.Array`):

            An array to store fragments in. Should be of ``dtype`` ``uint64``.

        num_workers (``int``):

            The number of parallel workers.

        mask (`class:daisy.Array`):

            A dataset containing a mask. If given, fragments are only extracted
            for masked-in (==1) areas.

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.

        epsilon_agglomerate (``float``):

            Perform an initial waterz agglomeration on the extracted fragments
            to this threshold. Skip if 0 (default).

        filter_fragments (``float``):

            Filter fragments that have an average affinity lower than this
            value.

        replace_sections (``list`` of ``int``):

            Replace fragments data with zero in given sections (useful if large
            artifacts are causing issues). List of section numbers (in voxels)
"""
    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    # prepare fragments dataset
    fragments = prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        # total_roi,
        affs.voxel_size,
        np.uint64,
        Roi((0, 0, 0), block_size),
        compressor='default')


if __name__ == '__main__':
    pass
