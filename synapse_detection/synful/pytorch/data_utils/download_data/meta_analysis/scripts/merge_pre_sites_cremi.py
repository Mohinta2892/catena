import h5py
import numpy as np
import os
import glob
from pathlib import Path
from collections import defaultdict

def merge_presynaptic_sites(input_path, output_path, xy_threshold_nm=100, z_threshold_nm=80):
    """
    Merge multiple presynaptic sites that are close together into single sites.
    This converts many-to-many connections to 1-to-many connections.
    
    Algorithm:
    1. Group presynaptic sites first by Z-plane (within z_threshold)
    2. Within each Z-group, cluster by XY proximity (within xy_threshold)
    3. For each cluster, create a single representative presynaptic site (centroid)
    4. Update partner relationships to point to the merged site
    
    Z-aware grouping prevents merging synapses from different neurons across slices.
    """
    print(f"\nPROCESSING: {Path(input_path).name}")
    
    with h5py.File(input_path, 'r') as f_in:
        if 'annotations/locations' not in f_in or 'annotations/presynaptic_site/partners' not in f_in:
            print("  No annotations or partners found")
            return
        
        locs = f_in['annotations/locations'][...]
        ann_ids = f_in['annotations/ids'][...]
        ann_types = f_in['annotations/types'][...]
        partners = f_in['annotations/presynaptic_site/partners'][...]
        resolution = np.array(f_in['annotations'].attrs['resolution'])
        
        print(f"  Original: {len(locs)} annotations, {len(partners)} connections")
        
        # Identify presynaptic and postsynaptic sites
        # Handle both bytes and string types
        ann_types_str = []
        for t in ann_types:
            if isinstance(t, bytes):
                ann_types_str.append(t.decode('utf-8'))
            else:
                ann_types_str.append(str(t))
        
        # Match actual annotation types - could be 'pres'/'post' or 'presynaptic_site'/'postsynaptic_site'
        pre_mask = np.array([t in ['pres', 'presynaptic_site'] for t in ann_types_str])
        pre_ids = set(ann_ids[pre_mask])
        
        # Build a map: presynaptic_id -> list of postsynaptic_ids
        pre_to_posts = defaultdict(list)
        for pre, post in partners:
            if pre in pre_ids:
                pre_to_posts[pre].append(post)
        
        print(f"  Presynaptic sites: {len(pre_ids)}")
        print(f"  Postsynaptic sites: {np.sum(~pre_mask)}")
        
        if len(pre_ids) == 0:
            print("  ⚠️  No presynaptic sites found! Check annotation types.")
            print(f"  All annotation types: {set(ann_types_str)}")
            return
        print(f"  XY threshold: {xy_threshold_nm} nm")
        print(f"  Z threshold: {z_threshold_nm} nm")
        
        # Group presynaptic sites by proximity (Z-aware)
        pre_indices = np.where(pre_mask)[0]
        pre_locs = locs[pre_indices]  # ZYX format
        pre_id_list = ann_ids[pre_indices]
        
        # Step 1: Group by Z-plane first
        z_groups = defaultdict(list)
        for i, (pre_id, pre_loc) in enumerate(zip(pre_id_list, pre_locs)):
            z_coord = pre_loc[0]  # Z is first dimension
            # Assign to Z-group (round to nearest z_threshold)
            z_bin = int(z_coord / z_threshold_nm)
            z_groups[z_bin].append((i, pre_id, pre_loc))
        
        print(f"  Z-plane groups: {len(z_groups)}")
        
        # Step 2: Within each Z-group, cluster by XY proximity
        merged_groups = []
        total_pre_sites = 0
        
        for z_bin, z_group in z_groups.items():
            if len(z_group) == 0:
                continue
            
            total_pre_sites += len(z_group)
            used = set()
            
            for i, pre_id, pre_loc in z_group:
                if pre_id in used:
                    continue
                
                # Start a new cluster in this Z-plane
                cluster = [(pre_id, pre_loc)]
                used.add(pre_id)
                
                # Find nearby presynaptic sites in XY within same Z-group
                for j, other_id, other_loc in z_group:
                    if other_id in used:
                        continue
                    
                    # Check Z distance first (must be within Z threshold)
                    z_dist = abs(other_loc[0] - pre_loc[0])
                    if z_dist > z_threshold_nm:
                        continue
                    
                    # Check XY distance to any member of current cluster
                    for _, cluster_loc in cluster:
                        xy_dist = np.linalg.norm(other_loc[1:] - cluster_loc[1:])  # YX distance
                        
                        if xy_dist < xy_threshold_nm:
                            # Also verify Z compatibility
                            z_dist_to_cluster = abs(other_loc[0] - cluster_loc[0])
                            if z_dist_to_cluster <= z_threshold_nm:
                                cluster.append((other_id, other_loc))
                                used.add(other_id)
                                break
                
                merged_groups.append(cluster)
        
        print(f"  Merged {total_pre_sites} presynaptic sites into {len(merged_groups)} clusters")
        
        # Statistics on cluster sizes
        if len(merged_groups) > 0:
            cluster_sizes = [len(g) for g in merged_groups]
            print(f"  Cluster size stats: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                  f"mean={np.mean(cluster_sizes):.1f}")
            print(f"  Single-site clusters: {sum(1 for s in cluster_sizes if s == 1)}")
            print(f"  Multi-site clusters: {sum(1 for s in cluster_sizes if s > 1)}")
        else:
            print(f"  ⚠️  No clusters created!")
            return
        
        # Create new annotations
        new_locs = []
        new_ids = []
        new_types = []
        new_partners = []
        
        # Map old pre IDs to new merged IDs
        old_to_new_pre_id = {}
        next_id = int(np.max(ann_ids)) + 1
        
        # Add merged presynaptic sites
        for cluster in merged_groups:
            # Compute centroid
            cluster_locs = np.array([loc for _, loc in cluster])
            centroid = np.mean(cluster_locs, axis=0)
            
            # Create new merged ID
            merged_id = next_id
            next_id += 1
            
            # Map all old IDs to new merged ID
            for old_id, _ in cluster:
                old_to_new_pre_id[old_id] = merged_id
            
            new_locs.append(centroid)
            new_ids.append(merged_id)
            new_types.append(b'presynaptic_site')  # Convert to full CREMI format
        
        # Add postsynaptic sites (unchanged locations, but convert type to full format)
        post_indices = np.where(~pre_mask)[0]
        for idx in post_indices:
            new_locs.append(locs[idx])
            new_ids.append(ann_ids[idx])
            new_types.append(b'postsynaptic_site')  # Convert to full CREMI format
        
        # Update partner relationships
        for old_pre_id, posts in pre_to_posts.items():
            new_pre_id = old_to_new_pre_id[old_pre_id]
            for post_id in posts:
                new_partners.append([new_pre_id, post_id])
        
        # Remove duplicate connections (same pre and post)
        new_partners = np.array(new_partners)
        if len(new_partners) > 0:
            new_partners = np.unique(new_partners, axis=0)
        
        print(f"  New: {len(new_locs)} annotations, {len(new_partners)} unique connections")
        
        # Count connections per presynaptic site
        if len(new_partners) > 0:
            unique_pres, counts = np.unique(new_partners[:, 0], return_counts=True)
            avg_connections = np.mean(counts)
            max_connections = np.max(counts)
            print(f"  Avg connections per pre-site: {avg_connections:.1f}")
            print(f"  Max connections per pre-site: {max_connections}")
        
        # Write output
        with h5py.File(output_path, 'w') as f_out:
            # Copy volumes
            f_in.copy('volumes', f_out)
            
            # Write merged annotations
            ann_group = f_out.create_group('annotations')
            ann_group.attrs['resolution'] = resolution
            ann_group.attrs['offset'] = f_in['annotations'].attrs['offset']
            
            new_locs = np.array(new_locs)
            new_ids = np.array(new_ids)
            new_types = np.array(new_types)  # Already bytes format
            
            ann_group.create_dataset('ids', data=new_ids)
            ann_group.create_dataset('locations', data=new_locs.astype(np.float64))
            ann_group.create_dataset('types', data=new_types)
            
            if len(new_partners) > 0:
                pre_group = ann_group.create_group('presynaptic_site')
                pre_group.create_dataset('partners', data=new_partners.astype(np.uint64))
            
            # Copy comments
            if 'comments' in f_in:
                f_in.copy('comments', f_out)
            else:
                com_group = f_out.create_group('comments')
                com_group.create_dataset('target_ids', shape=(0,), dtype=np.uint64)
                com_group.create_dataset('comments', shape=(0,), dtype=h5py.special_dtype(vlen=str))
    
    print(f"  ✓ Saved: {output_path}\n")

def batch_process(input_dir, output_dir, xy_threshold_nm=100, z_threshold_nm=80):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = glob.glob(os.path.join(input_dir, "*.hdf"))
    print(f"Found {len(files)} files to process")
    print(f"XY threshold: {xy_threshold_nm} nm")
    print(f"Z threshold: {z_threshold_nm} nm\n")
    
    for f in files:
        out = os.path.join(output_dir, os.path.basename(f))
        merge_presynaptic_sites(f, out, xy_threshold_nm, z_threshold_nm)

if __name__ == "__main__":
    INPUT_DIR = "/media/samia/DATA/mounts/fibserver1/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/TEM/ORIGINAL/CREMI"
    OUTPUT_DIR = "/media/samia/DATA/mounts/fibserver1/samia_dani/SynapseDetectionPaper/data/COMBINED_DATASETS/TEM/ORIGINAL/CREMI_MERGED"
    
    # XY threshold: merge presynaptic sites within this XY distance (nm)
    # At 4nm XY resolution, 100nm = 25 pixels
    XY_THRESHOLD = 120
    
    # Z threshold: only merge sites within this Z distance (nm)
    # At 40nm Z resolution, 80nm = 2 slices
    # This prevents merging synapses from different neurons across Z slices
    Z_THRESHOLD = 80
    
    batch_process(INPUT_DIR, OUTPUT_DIR, XY_THRESHOLD, Z_THRESHOLD)
