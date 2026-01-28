import h5py
import numpy as np
import os
import glob
from pathlib import Path

def adjust_coordinates(input_path, output_path):
    """
    Convert annotation coordinates from relative to absolute by adding offsets.
    Set all offsets to [0,0,0]. No cropping of volumes.
    """
    print(f"\nPROCESSING: {Path(input_path).name}")
    
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            # Copy raw volume as-is, reset offset
            vol_group = f_out.create_group('volumes')
            raw_data = f_in['volumes/raw'][...]
            resolution = np.array(f_in['volumes/raw'].attrs['resolution'])
            
            raw_out = vol_group.create_dataset('raw', data=raw_data, chunks=True, compression='gzip')
            raw_out.attrs['resolution'] = resolution
            raw_out.attrs['offset'] = np.array([0.0, 0.0, 0.0])
            
            print(f"  Raw: {raw_data.shape}, resolution: {resolution}")
            
            # Copy labels, adjust their offsets to absolute positions
            labels_group = vol_group.create_group('labels')
            for label_name in ['neuron_ids', 'clefts', 'presynaptic_segments', 'postsynaptic_segments']:
                if f'volumes/labels/{label_name}' in f_in:
                    label_data = f_in[f'volumes/labels/{label_name}'][...]
                    label_offset = np.array(f_in[f'volumes/labels/{label_name}'].attrs.get('offset', [0, 0, 0]))
                    
                    label_out = labels_group.create_dataset(label_name, data=label_data, 
                                                           chunks=True, compression='gzip')
                    label_out.attrs['resolution'] = resolution
                    # Keep the offset so labels align correctly with raw
                    label_out.attrs['offset'] = label_offset
                    
                    print(f"  {label_name}: {label_data.shape}, offset: {label_offset}")
            
            # Adjust annotations: add offset to locations, set offset to 0
            ann_group = f_out.create_group('annotations')
            ann_group.attrs['resolution'] = resolution
            ann_group.attrs['offset'] = np.array([0.0, 0.0, 0.0])
            
            if 'annotations/locations' in f_in:
                locs = f_in['annotations/locations'][...]
                ann_offset = np.array(f_in['annotations'].attrs.get('offset', [0, 0, 0]))
                ann_ids = f_in['annotations/ids'][...]
                ann_types = f_in['annotations/types'][...]
                
                # Transform: new_location = old_location + offset
                new_locs = locs + ann_offset
                
                # Check bounds against raw volume
                raw_size_nm = np.array(raw_data.shape) * resolution
                valid_mask = (
                    np.all(new_locs >= 0, axis=1) & 
                    np.all(new_locs < raw_size_nm, axis=1)
                )
                
                n_invalid = np.sum(~valid_mask)
                if n_invalid > 0:
                    print(f"  ⚠️  Filtering {n_invalid}/{len(locs)} synapses outside raw volume bounds")
                    print(f"    Raw volume size: {raw_size_nm} nm")
                    print(f"    Synapse range: {np.min(new_locs, axis=0)} to {np.max(new_locs, axis=0)}")
                
                new_locs = new_locs[valid_mask]
                ann_ids = ann_ids[valid_mask]
                ann_types = ann_types[valid_mask]
                
                print(f"  Annotations: {len(locs)} → {len(new_locs)}")
                print(f"    Old offset: {ann_offset}")
                print(f"    New range: {np.min(new_locs, axis=0)} to {np.max(new_locs, axis=0)}")
                
                # Filter partners to only include valid IDs
                if 'annotations/presynaptic_site/partners' in f_in:
                    partners = f_in['annotations/presynaptic_site/partners'][...]
                    valid_ids_set = set(ann_ids)
                    valid_partners_mask = np.array([
                        (pre in valid_ids_set and post in valid_ids_set) 
                        for pre, post in partners
                    ])
                    partners = partners[valid_partners_mask]
                    print(f"    Partners: {len(f_in['annotations/presynaptic_site/partners'][...])} → {len(partners)}")
                else:
                    partners = None
                
                ann_group.create_dataset('ids', data=ann_ids)
                ann_group.create_dataset('locations', data=new_locs.astype(np.float64))
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
        adjust_coordinates(f, out)

if __name__ == "__main__":
    INPUT_DIR = "/media/samia/DATA/mounts/fibserver1/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/TEM/OLD/FAFB"
    OUTPUT_DIR = "/media/samia/DATA/mounts/fibserver1/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/TEM/OLD/FAFB_ADJUSTED"
    
    batch_process(INPUT_DIR, OUTPUT_DIR)
