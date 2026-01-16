import os
import argparse
import pickle
import numpy as np
import zarr
import kimimaro
from tqdm import tqdm


def run_kimimaro_skeletonization(zarr_file, dataset, anisotropy=None, dust_threshold=1000):
    """
    Run kimimaro skeletonization on a zarr file's segmentation dataset.

    Parameters:
    -----------
    zarr_file : str
        Path to the zarr file
    dataset : str
        Name of the segmentation dataset within the zarr file
    anisotropy : tuple, optional
        Voxel anisotropy as (z, y, x), default is None (isotropic)
    dust_threshold : int, optional
        Skip components with fewer than this many voxels, default is 1000

    Returns:
    --------
    dict
        Dictionary of skeletons keyed by segment ID
    """

    if zarr_file.endswith('.zarr'):
        # Load segmentation data from zarr
        print(f"Loading segmentation from {zarr_file}, dataset: {dataset}")
        seg = zarr.open(zarr_file, mode='r')[dataset][...]
    elif zarr_file.endswith((".npy", ".npz")):
        print(f"Loading segmentation from {zarr_file}")
        seg = np.load(zarr_file)

    print(seg.shape)
    # Run skeletonization
    print(f"Running kimimaro skeletonization...")
    skeletons = kimimaro.skeletonize(
        seg,
        teasar_params={
            "scale": 1.0,
            "const": 500,
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
        },
        dust_threshold=dust_threshold,
        anisotropy=anisotropy,
        fix_branching=True,
        fix_borders=True,
        progress=True,
        parallel=1
    )

    print(f"Skeletonization complete. Generated {len(skeletons)} skeletons.")
    return skeletons, seg


def verify_skeleton_graph(skel_path, seg):
    """
    Verify that the saved skeleton is a valid network graph by checking
    if it has the expected attributes and structure.

    Parameters:
    -----------
    skel_path : str
        Path to the saved skeleton pickle file
    seg : np.ndarray
        Original segmentation array used to create the skeleton

    Returns:
    --------
    bool
        True if verification passes, False otherwise
    """
    try:
        print(f"Verifying skeleton at {skel_path}...")
        with open(skel_path, "rb") as f:
            skels = pickle.load(f)

        if not isinstance(skels, dict):
            print("ERROR: Skeleton is not a dictionary of segment IDs to skeleton objects")
            return False

        if len(skels) == 0:
            print("WARNING: No skeletons were generated")
            return False

        # Check the first skeleton
        first_id = next(iter(skels))
        skel = skels[first_id]

        # Check if it has nodes attribute
        if not hasattr(skel, 'nodes'):
            print("ERROR: Skeleton does not have 'nodes' attribute")
            return False

        # Check if nodes have index_position attribute
        if len(skel.nodes) == 0:
            print("WARNING: Skeleton has no nodes")
            return False

        first_node = next(iter(skel.nodes))
        if 'index_position' not in skel.nodes[first_node]:
            print("ERROR: Nodes do not have 'index_position' attribute")
            return False

        # Try to access a node position and verify it's within the segmentation bounds
        x, y, z = skel.nodes[first_node]['index_position']
        if x >= seg.shape[0] or y >= seg.shape[1] or z >= seg.shape[2]:
            print(f"ERROR: Node position ({x},{y},{z}) is outside segmentation bounds {seg.shape}")
            return False

        # Try to assign pred_id to verify the expected usage pattern works
        try:
            for node in tqdm(list(skel.nodes)[:5]):  # Just check a few nodes for speed
                x, y, z = skel.nodes[node]["index_position"]
                skel.nodes[node]["pred_id"] = seg[x, y, z]
            print("Successfully assigned pred_id to nodes")
        except Exception as e:
            print(f"ERROR: Failed to assign pred_id to nodes: {e}")
            return False

        print("Skeleton verification PASSED!")
        return True

    except Exception as e:
        print(f"ERROR during skeleton verification: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Skeletonize segmentation data from zarr files')
    parser.add_argument('filename', help='Path to zarr file containing segmentation')
    parser.add_argument('dataset', help='Dataset name within zarr file containing segmentation')
    parser.add_argument('--anisotropy', nargs=3, type=float, metavar=('Z', 'Y', 'X'),
                        help='Voxel anisotropy as Z Y X (default: None, isotropic)')
    parser.add_argument('--dust-threshold', type=int, default=1000,
                        help='Skip components with fewer than this many voxels (default: 1000)')
    parser.add_argument('--skip-verification', action='store_true',
                        help='Skip skeleton verification step')

    args = parser.parse_args()

    # Get output filename (same location as input, but with .skel.pkl extension)
    output_file = os.path.splitext(args.filename)[0] + '.skel.pkl'

    # Run skeletonization
    skeletons, seg = run_kimimaro_skeletonization(
        args.filename,
        args.dataset,
        anisotropy=tuple(args.anisotropy) if args.anisotropy else None,
        dust_threshold=args.dust_threshold
    )

    # Save skeletons
    print(f"Saving skeletons to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(skeletons, f)

    # Verify the saved skeleton
    if not args.skip_verification:
        if verify_skeleton_graph(output_file, seg):
            print("Skeleton verification successful!")
        else:
            print("Skeleton verification failed!")

    print("Done!")


if __name__ == "__main__":
    main()