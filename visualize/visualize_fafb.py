"""
Neuroglancer visualization for FAFB data with proper offset handling.

This script visualizes the ORIGINAL (uncropped) volumes and shows where
annotations are located when accounting for their offsets.

Key points:
- Volumes have an offset attribute indicating where they start in global space
- Annotations have locations that are RELATIVE to the annotation offset
- Global position = annotation_offset + annotation_location

Author: Modified for FAFB offset verification
"""

import argparse
import itertools
import os
import h5py
import neuroglancer
import numpy as np

ngid = itertools.count(start=1)


def load_hdf5_with_offset(filename, dataset):
    """Load HDF5 dataset and its offset attribute."""
    f = h5py.File(filename, 'r')
    offset = np.array([0.0, 0.0, 0.0])
    
    if dataset in f:
        data = f[dataset][:]
        if 'offset' in f[dataset].attrs.keys():
            offset = np.array(f[dataset].attrs['offset'])
        print(f"{dataset}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Offset (nm): {offset}")
    else:
        data = None
        print(f"{dataset} does not exist")
    
    f.close()
    return data, offset


def verify_annotations(filename):
    """
    Verify that annotations fall within the volume bounds.
    This helps us understand if the offset handling is correct.
    """
    print("\n" + "="*80)
    print("VERIFICATION: Checking annotation bounds")
    print("="*80)
    
    with h5py.File(filename, 'r') as f:
        # Load volume info
        raw_shape = np.array(f['volumes/raw'].shape)
        resolution = np.array(f['volumes/raw'].attrs['resolution'])
        vol_offset = np.array(f['volumes/raw'].attrs['offset'])
        
        vol_size_nm = raw_shape * resolution
        vol_global_start = np.array([0.0, 0.0, 0.0])  # Volume starts at origin
        vol_global_end = vol_size_nm
        
        print(f"\nVOLUME:")
        print(f"  Shape (voxels): {raw_shape}")
        print(f"  Resolution (nm/voxel): {resolution}")
        print(f"  Volume offset attribute (nm): {vol_offset}")
        print(f"  Volume size (nm): {vol_size_nm}")
        print(f"  Volume global span: {vol_global_start} to {vol_global_end}")
        
        # Load annotation info
        if 'annotations/locations' in f:
            locs_local = np.array(f['annotations/locations'][...])
            ann_offset = np.array(f['annotations'].attrs.get('offset', vol_offset))
            
            # Calculate global positions
            locs_global = ann_offset + locs_local
            
            min_local = np.min(locs_local, axis=0)
            max_local = np.max(locs_local, axis=0)
            min_global = np.min(locs_global, axis=0)
            max_global = np.max(locs_global, axis=0)
            
            print(f"\nANNOTATIONS:")
            print(f"  Count: {len(locs_local)}")
            print(f"  Annotation offset (nm): {ann_offset}")
            print(f"  Local coordinates (relative to offset):")
            print(f"    Min: {min_local}")
            print(f"    Max: {max_local}")
            print(f"  Global coordinates (offset + local):")
            print(f"    Min: {min_global}")
            print(f"    Max: {max_global}")
            
            # Check bounds
            print(f"\nBOUNDS CHECK:")
            if np.any(min_global < vol_global_start):
                print(f"  ⚠️  WARNING: Some annotations BELOW volume start!")
                print(f"     Min global: {min_global}")
                print(f"     Volume start: {vol_global_start}")
                print(f"     Undershoot: {vol_global_start - min_global}")
            else:
                print(f"  ✓ All annotations >= volume start")
            
            if np.any(max_global > vol_global_end):
                print(f"  ⚠️  WARNING: Some annotations BEYOND volume end!")
                print(f"     Max global: {max_global}")
                print(f"     Volume end: {vol_global_end}")
                print(f"     Overshoot: {max_global - vol_global_end}")
            else:
                print(f"  ✓ All annotations <= volume end")
            
            if np.all(min_global >= vol_global_start) and np.all(max_global <= vol_global_end):
                print(f"\n  ✓✓✓ ALL ANNOTATIONS ARE WITHIN VOLUME BOUNDS ✓✓✓")
            else:
                print(f"\n  ❌ SOME ANNOTATIONS ARE OUTSIDE VOLUME BOUNDS ❌")
                print(f"     This suggests the offset handling needs attention!")
        
        print("="*80 + "\n")


def add_fafb_synapses_with_offsets(s, filename, res):
    """
    Add synapse annotations to neuroglancer viewer.
    Annotations are RELATIVE to their offset - we just use them directly.
    """
    
    with h5py.File(filename, 'r') as f:
        locs_local = f['annotations/locations'][...]
        ann_offset = np.array(f['annotations'].attrs.get('offset', [0, 0, 0]))
        
        print(f"\nLOADING ANNOTATIONS:")
        print(f"  Count: {len(locs_local)}")
        print(f"  Annotation offset: {ann_offset}")
        print(f"  Using locations AS-IS (they're relative to label volume)")
        
        # Load partner relationships
        if 'annotations/presynaptic_site/partners' in f:
            partners = f['annotations/presynaptic_site/partners'][...]
            annotation_ids = f['annotations/ids'][...]
        else:
            partners = []
            annotation_ids = []
    
    # Create annotation lists
    pre_sites = []
    post_sites = []
    connectors = []
    
    for (pre, post) in partners:
        pre_indices = np.where(annotation_ids == pre)[0]
        post_indices = np.where(annotation_ids == post)[0]
        
        if len(pre_indices) == 0 or len(post_indices) == 0:
            continue
        
        # Use local coordinates directly (they match the label volume coordinate system)
        pre_site = locs_local[int(pre_indices[0])]
        post_site = locs_local[int(post_indices[0])]
        
        pre_sites.append(neuroglancer.PointAnnotation(point=pre_site, id=next(ngid)))
        post_sites.append(neuroglancer.PointAnnotation(point=post_site, id=next(ngid)))
        connectors.append(neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site, id=next(ngid)))
    
    print(f"  Created {len(pre_sites)} presynaptic sites")
    print(f"  Created {len(post_sites)} postsynaptic sites")
    print(f"  Created {len(connectors)} connectors")
    
    # Add layers to viewer
    s.layers.append(
        name="connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            annotations=connectors
        )
    )
    
    s.layers.append(
        name="pre_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#FF0000'  # Red for presynaptic
                )
            ],
            annotations=pre_sites
        )
    )
    
    s.layers.append(
        name="post_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#00FF00'  # Green for postsynaptic
                )
            ],
            annotations=post_sites
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FAFB data with proper offset handling"
    )
    parser.add_argument(
        '-sfile',
        required=True,
        help="Path to HDF5 file with FAFB data"
    )
    parser.add_argument(
        '-res',
        default="40,4,4",
        help="Resolution as Z,Y,X (e.g., '40,4,4')"
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help="Only run verification, don't start viewer"
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    res = tuple(int(x.strip()) for x in args.res.split(','))
    print(f"Using resolution: {res} nm/voxel (Z, Y, X)")
    
    # First, verify the data
    verify_annotations(args.sfile)
    
    if args.verify_only:
        print("Verification complete. Exiting (--verify-only flag set).")
        return
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA FOR VISUALIZATION")
    print("="*80 + "\n")
    
    raw, raw_offset = load_hdf5_with_offset(args.sfile, 'volumes/raw')
    neuron_ids, neuron_offset = load_hdf5_with_offset(args.sfile, 'volumes/labels/neuron_ids')
    clefts, cleft_offset = load_hdf5_with_offset(args.sfile, 'volumes/labels/clefts')
    
    # Labels are already cropped - their offset tells where they START in the raw volume
    # Convert nm offsets to voxel offsets
    raw_voxel_offset = tuple(int(o / r) for o, r in zip(raw_offset, res))
    
    print(f"\nVolume layer offsets (in voxels):")
    print(f"  Raw: {raw_voxel_offset}")
    
    if neuron_ids is not None:
        neuron_voxel_offset = tuple(int(o / r) for o, r in zip(neuron_offset, res))
        print(f"  Neuron IDs: {neuron_voxel_offset}")
    
    if clefts is not None:
        clefts[clefts == np.array(-1).astype(np.uint64)] = 0
        cleft_voxel_offset = tuple(int(o / r) for o, r in zip(cleft_offset, res))
        print(f"  Clefts: {cleft_voxel_offset}")
    
    # Create neuroglancer dimensions
    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units="nm",
        scales=res,
    )
    
    # Create viewer
    print("\n" + "="*80)
    print("STARTING NEUROGLANCER VIEWER")
    print("="*80 + "\n")
    
    viewer = neuroglancer.Viewer()
    
    with viewer.txn() as s:
        # Add raw data
        s.layers.append(
            name='raw',
            layer=neuroglancer.LocalVolume(
                raw,
                dimensions=dimensions,
                volume_type='image',
                voxel_offset=raw_voxel_offset
            )
        )
        
        # Add segmentation layers if available
        if neuron_ids is not None:
            s.layers.append(
                name='neuron_ids',
                layer=neuroglancer.LocalVolume(
                    neuron_ids,
                    dimensions=dimensions,
                    volume_type='segmentation',
                    voxel_offset=neuron_voxel_offset
                )
            )
        
        if clefts is not None:
            s.layers.append(
                name='clefts',
                layer=neuroglancer.LocalVolume(
                    clefts,
                    dimensions=dimensions,
                    volume_type='segmentation',
                    voxel_offset=cleft_voxel_offset
                )
            )
        
        # Add synapse annotations with proper offset handling
        add_fafb_synapses_with_offsets(s, args.sfile, res)
    
    print(f"\nNeuroglancer viewer ready at:")
    print(viewer)
    print("\nPress Ctrl+C to exit")
    
    # Keep the viewer alive
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == '__main__':
    main()
