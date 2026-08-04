"""
skeletonise.py — Fetch, skeletonise, and save neurons from a CAVE-backed segmentation volume.

Usage examples
--------------
# Single ID via CLI:
    python skeletonise_modular.py --ids 648518346358338307

# Multiple IDs via CLI:
    python skeletonise_modular.py --ids 648518346358338307 648518346362917146

# IDs from a CSV file (column named 'target_id' by default):
    python skeletonise_modular.py --csv neurons.csv

# CSV with a different ID column name:
    python skeletonise_modular.py --csv neurons.csv --csv-column segment_id

# Override any default parameters:
NB:  --coord-scale 64 is required to map the mip=3 to mip=0. if using mip=4, put --coord-scale 128. (8x2^4)

    python skeletonise_modular.py --ids 648518346358338307 \
        --datastack zlatic_mr143_datastack \
        --mip 3 \
        --const 10 \
        --pdrf-exponent 2 \
        --scale 1 \
        --smooth-window 2 \
        --resample 128 \
        --output-dir ./skeletons \
        --save-mesh \
        --mesh-dir ./meshes
        --coord-scale 64 
        

"""

import argparse
import os
import pickle
import sys

import kimimaro
import navis as nv
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from cloudvolume.lib import Bbox


# ---------------------------------------------------------------------------
# Configuration dataclass (plain dict works fine; swap for dataclasses.dataclass
# or pydantic.BaseModel if you want validation later)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # CAVE / cloud volume
    "datastack_name":"zlatic_octo-2_datastack",
    "server_address":"https://global.connectomics.braininbrain.org",
    # Skeletonisation
    "mip":              3,      # downsampling level fed to kimimaro
    "scale":            1,
    "const":            10,
    "pdrf_scale":       10000,
    "pdrf_exponent":    2,
    "pdrf_soma_acceptance_threshold": 30,
    "pdrf_soma_detection_threshold":  20,
    "pdrf_soma_invalidation_const":   30,
    "pdrf_soma_invalidation_scale":   1,
    "max_paths":        None,
    "dust_threshold":   0,
    "parallel":         2,
    # Post-processing
    "smooth_window":    2,
    "resample_to":      128,
    "coord_scale":      64,     # multiply raw skeleton coords (voxel → nm)
    # Output
    "output_dir":       "./skeletons",
    "save_mesh":        False,
    "mesh_dir":         "./meshes",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def initialise_cave_client(datastack_name: str, server_address: str) -> CAVEclient:
    """Initialise a CAVE client for the given datastack."""
    print(f"Initialising CAVE client  [{datastack_name}] ...")
    return CAVEclient(datastack_name=datastack_name, server_address=server_address)


def get_proofread_skeletons(cave: CAVEclient, search_ids: list, cfg: dict):
    """
    Download segmentation and skeletonise the requested segment IDs.

    Parameters
    ----------
    cave        : Initialised CAVEclient.
    search_ids  : List of integer segment IDs.
    cfg         : Config dict (uses keys: mip, scale, const, pdrf_*, max_paths,
                  dust_threshold, parallel).

    Returns
    -------
    dict  : {seg_id: kimimaro.Skeleton}
    """
    search_ids = [i for i in search_ids if i != 0]
    if not search_ids:
        raise ValueError("No valid (non-zero) segment IDs provided.")

    print("Loading segmentation volume ...")
    vol_seg = cave.info.segmentation_cloudvolume(
        agglomerate=True, mip=cfg["mip"]
    )
    bbox = Bbox([0, 0, 0], list(vol_seg.shape[:3]))

    print(f"Downloading segmentation for {len(search_ids)} ID(s) ...")
    seg_subset = vol_seg.download(bbox=bbox, agglomerate=False, segids=search_ids)
    seg_subset = seg_subset[:, :, :, 0]

    print("Skeletonising ...")
    skels = kimimaro.skeletonize(
        seg_subset,
        teasar_params={
            "scale":                        cfg["scale"],
            "const":                        cfg["const"],
            "pdrf_scale":                   cfg["pdrf_scale"],
            "pdrf_exponent":                cfg["pdrf_exponent"],
            "soma_acceptance_threshold":    cfg["pdrf_soma_acceptance_threshold"],
            "soma_detection_threshold":     cfg["pdrf_soma_detection_threshold"],
            "soma_invalidation_const":      cfg["pdrf_soma_invalidation_const"],
            "soma_invalidation_scale":      cfg["pdrf_soma_invalidation_scale"],
            "max_paths":                    cfg["max_paths"],
        },
        dust_threshold=cfg["dust_threshold"],
        fix_branching=True,
        fix_borders=True,
        fill_holes=False,
        fix_avocados=False,
        progress=True,
        parallel=cfg["parallel"],
        parallel_chunk_size=1,
    )
    return skels


def fetch_and_save_mesh(vol, target_id: int, name: str, mesh_dir: str):
    """Download the mesh for *target_id*, save as STL and pickle."""
    os.makedirs(mesh_dir, exist_ok=True)
    print(f"  Fetching mesh for {target_id} ...")
    m = vol.mesh.get([target_id], as_navis=True)
    neuron = m[0]
    neuron.name = name
    stl_path = os.path.join(mesh_dir, f"{name}.stl")
    pkl_path  = os.path.join(mesh_dir, f"{name}.pkl")
    nv.write_mesh(neuron, stl_path)
    with open(pkl_path, "wb") as fh:
        pickle.dump(neuron, fh)
    print(f"  Mesh saved → {stl_path}, {pkl_path}")


def postprocess_skeleton(raw_neuron, cfg: dict):
    """Scale, heal, smooth, and resample a raw navis skeleton neuron."""
    scaled    = raw_neuron * cfg["coord_scale"]
    healed    = nv.heal_skeleton(scaled)
    smoothed  = nv.smooth_skeleton(healed, window=cfg["smooth_window"])
    resampled = smoothed.resample(resample_to=cfg["resample_to"])
    return resampled


def process_single_id(target_id: int, name: str, skeletons: dict, vol, cfg: dict):
    """
    Post-process a single neuron's skeleton (and optionally its mesh) and
    write the result to disk.

    Parameters
    ----------
    target_id : Segment ID.
    name      : Human-readable label (used in file names).
    skeletons : Full kimimaro output dict — the entry for *target_id* is used.
    vol       : CloudVolume instance (needed only when save_mesh is True).
    cfg       : Config dict.
    """
    os.makedirs(cfg["output_dir"], exist_ok=True)

    if cfg["save_mesh"]:
        fetch_and_save_mesh(vol, target_id, name, cfg["mesh_dir"])

    if target_id not in skeletons:
        print(f"  WARNING: no skeleton found for {target_id} — skipping.")
        return

    print(f"  Post-processing skeleton for {target_id} ...")
    raw_nl   = nv.NeuronList(
        [nv.read_swc(s.to_swc(), id=i) for i, s in skeletons.items()
         if i == target_id]
    )
    processed = postprocess_skeleton(raw_nl[0], cfg)
    processed.id   = target_id
    processed.name = name

    out_path = os.path.join(cfg["output_dir"], f"{target_id}.npy")
    np.save(out_path, processed, allow_pickle=True)
    print(f"  Skeleton saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_ids_from_csv(csv_path: str, column: str) -> list[int]:
    """Read segment IDs from *column* of a CSV file."""
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[column].dropna().astype(int).tolist()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Skeletonise neurons from a CAVE segmentation volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- ID input (mutually exclusive) ---
    id_group = p.add_mutually_exclusive_group(required=True)
    id_group.add_argument(
        "--ids", nargs="+", type=int, metavar="ID",
        help="One or more segment IDs.",
    )
    id_group.add_argument(
        "--csv", metavar="FILE",
        help="Path to a CSV file containing segment IDs.",
    )
    p.add_argument(
        "--csv-column", default="target_id", metavar="COL",
        help="Column name inside the CSV that holds the segment IDs.",
    )
    p.add_argument(
        "--names-column", default=None, metavar="COL",
        help="Optional CSV column with human-readable names for each ID.",
    )

    # --- CAVE / volume ---
    p.add_argument("--datastack",      default=DEFAULT_CONFIG["datastack_name"])
    p.add_argument("--server-address", default=DEFAULT_CONFIG["server_address"])

    # --- Skeletonisation ---
    p.add_argument("--mip",           type=int,   default=DEFAULT_CONFIG["mip"],
                   help="MIP level (downsampling factor) for segmentation download.")
    p.add_argument("--scale",         type=float, default=DEFAULT_CONFIG["scale"])
    p.add_argument("--const",         type=float, default=DEFAULT_CONFIG["const"])
    p.add_argument("--pdrf-exponent", type=int,   default=DEFAULT_CONFIG["pdrf_exponent"])
    p.add_argument("--max-paths",     type=int,   default=None,
                   help="Maximum TEASAR paths per object (None = unlimited).")
    p.add_argument("--parallel",      type=int,   default=DEFAULT_CONFIG["parallel"],
                   help="kimimaro parallel workers (0 = all CPUs).")

    # --- Post-processing ---
    p.add_argument("--smooth-window", type=int,   default=DEFAULT_CONFIG["smooth_window"])
    p.add_argument("--resample",      type=int,   default=DEFAULT_CONFIG["resample_to"],
                   metavar="NM", help="Target inter-node spacing after resampling (nm).")
    p.add_argument("--coord-scale",   type=float, default=DEFAULT_CONFIG["coord_scale"],
                   help="Multiply raw skeleton coordinates by this factor (voxel → nm).")

    # --- Output ---
    p.add_argument("--output-dir",    default=DEFAULT_CONFIG["output_dir"],
                   help="Directory for saved .npy skeletons.")
    p.add_argument("--save-mesh",     action="store_true",
                   help="Also download and save mesh (STL + pickle).")
    p.add_argument("--mesh-dir",      default=DEFAULT_CONFIG["mesh_dir"],
                   help="Directory for saved mesh files.")

    return p


def args_to_config(args: argparse.Namespace) -> dict:
    """Merge CLI arguments into a config dict."""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "datastack_name": args.datastack,
        "server_address": args.server_address,
        "mip":            args.mip,
        "scale":          args.scale,
        "const":          args.const,
        "pdrf_exponent":  args.pdrf_exponent,
        "max_paths":      args.max_paths,
        "parallel":       args.parallel,
        "smooth_window":  args.smooth_window,
        "resample_to":    args.resample,
        "coord_scale":    args.coord_scale,
        "output_dir":     args.output_dir,
        "save_mesh":      args.save_mesh,
        "mesh_dir":       args.mesh_dir,
    })
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args   = parser.parse_args()
    cfg    = args_to_config(args)

    # --- Collect IDs (and optional names) ---
    if args.csv:
        df = pd.read_csv(args.csv)
        col = args.csv_column
        if col not in df.columns:
            parser.error(
                f"Column '{col}' not found in {args.csv}. "
                f"Available: {list(df.columns)}"
            )
        target_ids = df[col].dropna().astype(int).tolist()
        if args.names_column and args.names_column in df.columns:
            names = df[args.names_column].astype(str).tolist()
        else:
            names = [str(i) for i in target_ids]
    else:
        target_ids = args.ids
        names      = [str(i) for i in target_ids]

    if not target_ids:
        sys.exit("No segment IDs to process — exiting.")

    print(f"Processing {len(target_ids)} neuron(s): {target_ids}")

    # --- Initialise clients ---
    cave = initialise_cave_client(cfg["datastack_name"], cfg["server_address"])
    nv.patch_cloudvolume()
    vol  = cave.info.segmentation_cloudvolume()

    # --- Skeletonise all IDs in one batch (more efficient) ---
    skeletons = get_proofread_skeletons(cave, search_ids=target_ids, cfg=cfg)

    # --- Per-neuron post-processing and saving ---
    for target_id, name in zip(target_ids, names):
        print(f"\n{'─'*60}")
        print(f"  ID: {target_id}   name: {name}")
        process_single_id(target_id, name, skeletons, vol, cfg)

    print(f"\n{'─'*60}")
    print("Done.")


if __name__ == "__main__":
    main()
