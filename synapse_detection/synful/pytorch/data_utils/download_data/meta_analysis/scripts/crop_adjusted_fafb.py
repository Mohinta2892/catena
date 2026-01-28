import h5py
import numpy as np
import os
import glob
from pathlib import Path

def crop_to_labels(input_path, output_path):
    """
    Crop the adjusted data (where annotations are already in absolute coords with offset=0)
    to the bounding box of the label volumes. Adjust annotation coordinates to the new local space.
    """
    print(f"\nPROCESSING: {Path(input_path).name}")
    
    with h5py.File(input_path, 'r') as f_in:
        # Get raw volume info
        raw_data = f_in['volumes/raw'][...]
        raw_shape = np.array(raw_data.shape)
        resolution = np.array(f_in['volumes/raw'].attrs['resolution'])
        
        print(f"  Raw: {raw_shape}")
        
        # Find the label bounding box from any available label
        label_offset_nm = None
        label_shape = None
        
        for label_name in ['neuron_ids', 'clefts']:
            if f'volumes/labels/{label_name}' in f_in:
                label_shape = np.array(f_in[f'volumes/labels/{label_name}'].shape)
                label_offset_nm = np.array(f_in[f'volumes/labels/{label_name}'].attrs.get('offset', [0, 0, 0]))
                print(f"  Label ({label_name}): {label_shape}, offset: {label_offset_nm}")
                break
        
        if label_offset_nm is None or np.all(label_offset_nm == 0):
            print("  No label offset found - data may already be cropped")
            return
        
        # Calculate crop region in voxels
        crop_start_voxels = (label_offset_nm / resolution).astype(int)
        crop_end_voxels = crop_start_voxels + label_shape
        
        print(f"  Crop region: [{crop_start_voxels[0]}:{crop_end_voxels[0]}, "
              f"{crop_start_voxels[1]}:{crop_end_voxels[1]}, {crop_start_voxels[2]}:{crop_end_voxels[2]}]")
        
        # Load annotations (already in absolute coordinates)
        if 'annotations/locations' in f_in:
            locs_abs = f_in['annotations/locations'][...]
            ann_ids = f_in['annotations/ids'][...]
            ann_types = f_in['annotations/types'][...]
            
            # Convert to new local coordinate system (subtract the crop start in nm)
            crop_start_nm = crop_start_voxels * resolution
            locs_local = locs_abs - crop_start_nm
            
            # Filter to keep only annotations within the cropped volume
            cropped_vol_size_nm = label_shape * resolution
            valid_mask = (
                np.all(locs_local >= 0, axis=1) & 
                np.all(locs_local < cropped_vol_size_nm, axis=1)
            )
            
            n_filtered = np.sum(~valid_mask)
            if n_filtered > 0:
                print(f"  ⚠️  Filtering {n_filtered}/{len(locs_abs)} annotations outside crop region")
            
            locs_local = locs_local[valid_mask]
            ann_ids = ann_ids[valid_mask]
            ann_types = ann_types[valid_mask]
            
            print(f"  Annotations: {len(locs_abs)} → {len(locs_local)}")
            print(f"    New local range: {np.min(locs_local, axis=0)} to {np.max(locs_local, axis=0)}")
            
            # Filter partners
            if 'annotations/presynaptic_site/partners' in f_in:
                partners = f_in['annotations/presynaptic_site/partners'][...]
                valid_ids_set = set(ann_ids)
                valid_partners = [(pre, post) for pre, post in partners 
                                 if pre in valid_ids_set and post in valid_ids_set]
                partners = np.array(valid_partners) if valid_partners else np.array([]).reshape(0, 2)
                print(f"    Partners: {len(f_in['annotations/presynaptic_site/partners'][...])} → {len(partners)}")
            else:
                partners = None
        else:
            locs_local = None
        
        # Write cropped data
        with h5py.File(output_path, 'w') as f_out:
            vol_group = f_out.create_group('volumes')
            
            # Crop raw volume
            cropped_raw = raw_data[
                crop_start_voxels[0]:crop_end_voxels[0],
                crop_start_voxels[1]:crop_end_voxels[1],
                crop_start_voxels[2]:crop_end_voxels[2]
            ]
            
            print(f"  Cropped raw: {cropped_raw.shape} (should match {label_shape})")
            
            raw_out = vol_group.create_dataset('raw', data=cropped_raw, chunks=True, compression='gzip')
            raw_out.attrs['resolution'] = resolution
            raw_out.attrs['offset'] = np.array([0.0, 0.0, 0.0])
            
            # Copy labels (reset offset to 0)
            labels_group = vol_group.create_group('labels')
            for label_name in ['neuron_ids', 'clefts', 'presynaptic_segments', 'postsynaptic_segments']:
                if f'volumes/labels/{label_name}' in f_in:
                    label_data = f_in[f'volumes/labels/{label_name}'][...]
                    label_out = labels_group.create_dataset(label_name, data=label_data,
                                                           chunks=True, compression='gzip')
                    label_out.attrs['resolution'] = resolution
                    label_out.attrs['offset'] = np.array([0.0, 0.0, 0.0])
            
            # Save adjusted annotations
            ann_group = f_out.create_group('annotations')
            ann_group.attrs['resolution'] = resolution
            ann_group.attrs['offset'] = np.array([0.0, 0.0, 0.0])
            
            if locs_local is not None and len(locs_local) > 0:
                ann_group.create_dataset('ids', data=ann_ids)
                ann_group.create_dataset('locations', data=locs_local.astype(np.float64))
                ann_group.create_dataset('types', data=ann_types)
                
                if partners is not None and len(partners) > 0:
                    pre_group = ann_group.create_group('presynaptic_site')
                    pre_group.create_dataset('partners', data=partners)
            
            # Copy comments
            if 'comments' in f_in:
                f_in.copy('comments', f_out)
            else:
                com_group = f_out.create_group('comments')
                com_group.create_dataset('target_ids', shape=(0,), dtype=np.uint64)
                com_group.create_dataset('comments', shape=(0,), dtype=h5py.special_dtype(vlen=str))
    
    print(f"  ✓ Saved: {output_path}\n")

def batch_process(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = glob.glob(os.path.join(input_dir, "*.hdf"))
    print(f"Found {len(files)} files to process\n")
    
    for f in files:
        out = os.path.join(output_dir, os.path.basename(f))
        crop_to_labels(f, out)

if __name__ == "__main__":
    # This script operates on the ADJUSTED files (output of adjust_fafb_coordinates.py)
    INPUT_DIR = "/media/samia/DATA/mounts/fibserver1/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/TEM/OLD/FAFB_ADJUSTED"
    OUTPUT_DIR = "/media/samia/DATA/mounts/fibserver1/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/TEM/OLD/FAFB_CROPPED"
    
    batch_process(INPUT_DIR, OUTPUT_DIR)
