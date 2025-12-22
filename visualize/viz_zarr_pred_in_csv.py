"""
Visualize Zarr RAW data and CSV Synapse predictions.

Usage:
    python visualize_zarr_pred_in_csv.py <zarr_file> \
        --pre pred_pre_locations.csv \
        --post pred_post_locations.csv \
        --mapping pre_post_mapping.csv \
        --res 8 8 8
        
Example:
python /media/samia/DATA/PhD/codebases/restructured_packages/visualize/neuroglancer/viz_zarr_pred_in_csv_claude.py /media/samia/DATA/mounts/zstore1/catena/data/OCTO_AL/data_3d/test/octo_AL_crop.zarr --pre /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_alcrop_setup_03_octo_cube_all3_same_preid_256_300000/pred_pre_locations.csv --post /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_alcrop_setup_03_octo_cube_all3_same_preid_256_300000/pred_post_locations.csv --mapping /media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_alcrop_setup_03_octo_cube_all3_same_preid_256_300000/pre_post_mapping.csv --res 8 8 8

"""

import argparse
import itertools
import csv
import zarr
import neuroglancer
import numpy as np
import time

ngid = itertools.count(start=1)

def load_zarr(inputfilename, dataset):
    """Loads a dataset from a zarr file."""
    print(f"Opening Zarr: {inputfilename} [{dataset}]")
    try:
        f = zarr.open(inputfilename, 'r')
    except Exception as e:
        print(f"Error opening zarr file: {e}")
        return None, (0, 0, 0)

    offset = [0, 0, 0]
    if dataset in f:
        data = f[dataset]
        if 'offset' in f[dataset].attrs.keys():
            offset = list(f[dataset].attrs['offset'])
        
        # Load into memory
        data_np = data[:] 
        print(f"Loaded {dataset}: shape={data_np.shape}, dtype={data_np.dtype}, offset={offset} (nm)")
        return data_np, offset
    else:
        print(f"Dataset {dataset} does not exist in {inputfilename}")
        return None, [0, 0, 0]

def read_coordinates(csv_path, id_col, z_col, y_col, x_col):
    """
    Reads coordinates from a CSV.
    Returns coordinates in [Z, Y, X] order (nm units).
    """
    coords = {}
    print(f"Loading coordinates from {csv_path}...")
    try:
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Store as [Z, Y, X] in nanometers (as they are in the CSV)
                    coords[int(row[id_col])] = [
                        float(row[z_col]), 
                        float(row[y_col]),
                        float(row[x_col])
                    ]
                except (ValueError, KeyError) as e:
                    continue 
        print(f"Loaded {len(coords)} coordinates from {csv_path}")
    except FileNotFoundError:
        print(f"Warning: CSV file not found: {csv_path}")
    return coords

def add_csv_predictions(s, pre_csv, post_csv, map_csv):
    """
    Loads synapse predictions from CSVs and adds them as annotation layers.
    Coordinates in CSV are in nm and in Z,Y,X order.
    Annotations use 1:1:1 scale since coordinates are already in physical nm space.
    """
    # Read coordinates (in nm, Z,Y,X order)
    pre_locs = read_coordinates(pre_csv, 'Pre_ID', 'Pre_Z', 'Pre_Y', 'Pre_X')
    post_locs = read_coordinates(post_csv, 'Post_ID', 'Post_Z', 'Post_Y', 'Post_X')
    
    connectors = []
    pre_sites = []
    post_sites = []
    
    print(f"Loading mapping from {map_csv}...")
    try:
        with open(map_csv, mode='r') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                try:
                    pre_id = int(row['Pre_ID'])
                    post_id = int(row['Post_ID'])
                    
                    if pre_id in pre_locs and post_id in post_locs:
                        point_a = pre_locs[pre_id]  # [Z, Y, X] in nm
                        point_b = post_locs[post_id]  # [Z, Y, X] in nm
                        
                        # Add connector line
                        connectors.append(
                            neuroglancer.LineAnnotation(
                                point_a=point_a,
                                point_b=point_b,
                                id=next(ngid)
                            )
                        )
                        
                        # Add pre-synaptic point
                        if pre_id not in [p for p in range(len(pre_sites))]:
                            pre_sites.append(
                                neuroglancer.PointAnnotation(
                                    point=point_a,
                                    id=next(ngid)
                                )
                            )
                        
                        # Add post-synaptic point
                        if post_id not in [p for p in range(len(post_sites))]:
                            post_sites.append(
                                neuroglancer.PointAnnotation(
                                    point=point_b,
                                    id=next(ngid)
                                )
                            )
                        
                        count += 1
                except (ValueError, KeyError) as e:
                    continue
        print(f"Generated {count} synapse connections.")
    except FileNotFoundError:
        print(f"Warning: Mapping file not found: {map_csv}")
        return

    # Create annotation coordinate space with 1:1:1 scale
    # since CSV coordinates are already in absolute nm
    annotation_dims = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'], 
        units='nm', 
        scales=[1, 1, 1]
    )

    # Add connector lines
    s.layers.append(
        name="connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=annotation_dims,
            annotation_relationships=['connectors'],
            annotations=connectors,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ffff00',
                )
            ],
        )
    )
    
    # Add pre-synaptic sites
    s.layers.append(
        name="pre_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=annotation_dims,
            annotation_relationships=['connectors'],
            annotations=pre_sites,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#FF0000'
                )
            ],
        )
    )
    
    # Add post-synaptic sites
    s.layers.append(
        name="post_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=annotation_dims,
            annotation_relationships=['connectors'],
            annotations=post_sites,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ff00ff'
                )
            ],
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Zarr RAW data and CSV Synapse predictions.")
    parser.add_argument('file', type=str, help='Path to the Zarr file containing volumes/raw')
    parser.add_argument('--pre', type=str, required=True, help='Path to pred_pre_locations.csv')
    parser.add_argument('--post', type=str, required=True, help='Path to pred_post_locations.csv')
    parser.add_argument('--mapping', type=str, required=True, help='Path to pre_post_mapping.csv')
    parser.add_argument('--res', type=float, nargs='+', default=[8, 8, 8], 
                        help='Voxel resolution in nm (z, y, x)')

    args = parser.parse_args()

    # Parse resolution
    if len(args.res) == 3:
        res = args.res  # [z, y, x]
    else:
        print("Error: Resolution must have 3 values (z, y, x)")
        exit(1)

    # 1. Load Raw Data (Native Zarr is Z, Y, X)
    raw_data_path = 'volumes/raw'
    raw, r_offset_nm = load_zarr(args.file, raw_data_path)

    if raw is None:
        print("Could not load raw data. Exiting.")
        exit(1)

    print(f"Raw data shape (Z,Y,X): {raw.shape}")
    print(f"Raw data offset (Z,Y,X) nm: {r_offset_nm}")
    print(f"Resolution (Z,Y,X) nm: {res}")

    # 2. Create Neuroglancer Coordinate Space (Z, Y, X)
    dimensions = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'], 
        units='nm', 
        scales=res  # [z_res, y_res, x_res]
    )
    
    # Calculate voxel offset (convert nm offset to voxel coordinates)
    voxel_offset = [o / s for o, s in zip(r_offset_nm, res)]
    print(f"Voxel offset (Z,Y,X): {voxel_offset}")

    viewer = neuroglancer.Viewer()
    
    with viewer.txn() as s:
        # Add Raw Volume
        s.layers.append(
            name='raw', 
            layer=neuroglancer.LocalVolume(
                data=raw, 
                dimensions=dimensions,
                volume_type='image',
                voxel_offset=voxel_offset
            )
        )
        
        # Add Predictions (coordinates from CSV are already in nm, Z,Y,X)
        add_csv_predictions(s, args.pre, args.post, args.mapping)

        # 3. Auto-center camera
        shape = np.array(raw.shape)  # [Z, Y, X]
        scales = np.array(res)  # [z_res, y_res, x_res]
        off = np.array(r_offset_nm)  # [z_off, y_off, x_off]
        
        # Center is at offset + (shape * scales) / 2
        center_position = off + (shape * scales) / 2
        
        print(f"Centering camera at (Z,Y,X) nm: {center_position}")
        s.position = center_position.tolist()

    print(f"\nNeuroglancer viewer created: {viewer}")
    print("Keep this script running to maintain the viewer connection...")
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")
