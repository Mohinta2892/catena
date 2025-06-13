"""
This script provides tools to evaluate and potentially merge segments in a precomputed segmentation based on manually traced skeletons from CATMAID. It allows you to:

1. Load traced skeletons from CATMAID or from a CSV file
2. Load a precomputed segmentation from a Google Cloud bucket
3. Plot traced neuron arbors
4. Compare traced skeletons with segmentation to identify segments for merging
5. Plot putatively merged segments
6. Calculate metrics to evaluate recovery quality
7. Merge segments using caveclient

run in (beast): conda deactivate; conda activate ngl

Author: Samia Mohinta
Affiliation: Cambridge University, UK

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy import ndimage
from skimage import measure
import csv
import argparse
from typing import List, Dict, Tuple, Set, Optional, Union
import warnings
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Import optional packages with error handling
try:
    import pymaid

    PYMAID_AVAILABLE = True
except ImportError:
    PYMAID_AVAILABLE = False
    warnings.warn("pymaid not available. CATMAID functionality will be limited.")

try:
    from cloudvolume import CloudVolume

    CLOUDVOLUME_AVAILABLE = True
except ImportError:
    CLOUDVOLUME_AVAILABLE = False
    warnings.warn("cloud-volume not available. Precomputed segmentation loading will be limited.")

try:
    import caveclient

    CAVECLIENT_AVAILABLE = True
except ImportError:
    CAVECLIENT_AVAILABLE = False
    warnings.warn("caveclient not available. Merging functionality will be limited.")

try:
    import graph_tool.all as gt
    import funlib.evaluate

    GRAPHTOOL_AVAILABLE = True
except ImportError:
    GRAPHTOOL_AVAILABLE = False
    warnings.warn("graph-tool or funlib.evaluate not available. NERL metrics will be limited.")

DEBUG = False
DEBUG_N = 100


class NeuronRecovery:
    def __init__(self, gcloud_bucket: str = None, catmaid_credentials: Dict = None):
        """
        Initialize the NeuronRecovery class.
        
        Args:
            gcloud_bucket: URL to the Google Cloud bucket with precomputed segmentation
            catmaid_credentials: Dictionary with CATMAID credentials
        """
        self.gcloud_bucket = gcloud_bucket
        self.catmaid_credentials = catmaid_credentials
        self.traced_neurons = None
        self.segmentation = None
        self.segment_ids_map = {}
        self.merge_candidates = []

        # Connect to CATMAID if credentials provided
        if PYMAID_AVAILABLE and catmaid_credentials:
            try:
                self.catmaid_instance = pymaid.CatmaidInstance(
                    server=catmaid_credentials.get('server'),
                    api_token=catmaid_credentials.get('api_token'),
                    http_user=catmaid_credentials.get('http_user'),
                    http_password=catmaid_credentials.get('http_password')
                )
                pymaid.connect_catmaid(self.catmaid_instance)
                print("Connected to CATMAID server.")
            except Exception as e:
                print(f"Failed to connect to CATMAID: {e}")
                self.catmaid_instance = None
        else:
            self.catmaid_instance = None

        # Connect to cloud volume if URL provided
        if CLOUDVOLUME_AVAILABLE and gcloud_bucket:
            try:
                self.cv = CloudVolume(gcloud_bucket)
                print(f"Connected to Cloud Volume at {gcloud_bucket}")
            except Exception as e:
                print(f"Failed to connect to Cloud Volume: {e}")
                self.cv = None
        else:
            self.cv = None

    def load_skeletons_from_catmaid(self, skeleton_ids: List[int]) -> None:
        """
        Load skeletons from CATMAID using pymaid.
        
        Args:
            skeleton_ids: List of skeleton IDs to load
        """
        if not PYMAID_AVAILABLE:
            raise ImportError("pymaid is required to load skeletons from CATMAID")

        if not self.catmaid_instance:
            raise ValueError("No CATMAID connection established. Provide credentials first.")

        try:
            self.traced_neurons = pymaid.get_neuron(skeleton_ids)
            print(f"Loaded {len(self.traced_neurons)} neurons from CATMAID")
        except Exception as e:
            raise Exception(f"Failed to load skeletons from CATMAID: {e}")

    def load_skeletons_from_csv(self, csv_path: str, resolution: tuple = (1, 1, 1)) -> None:
        """
        Load skeletons from a CSV file with node coordinates.
        Expected format: skeleton_id, node_id, parent_id, x, y, z, ...
        
        Args:
            csv_path: Path to the CSV file
        """
        self.resolution = resolution
        self.skeleton_nodes_map = {}
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            required_columns = ['skeleton_id', 'node_id', 'parent_id', 'x', 'y', 'z']

            # Check if required columns exist
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV file must contain column: {col}")

            # Group by skeleton_id
            grouped = df.groupby('skeleton_id')
            neurons = []

            for skeleton_id, group in grouped:
                # Create a nodes DataFrame with the required structure
                nodes_df = group[['node_id', 'parent_id', 'x', 'y', 'z']].copy()

                # Maybe we don't need it here
                # if DEBUG:  # only use top 100 nodes
                #     nodes_df = nodes_df.iloc[:DEBUG_N]

                # make sure x,y,z are floats
                nodes_df[['x', 'y', 'z']] = nodes_df[['x', 'y', 'z']].apply(pd.to_numeric, errors='raise')
                print(self.resolution)
                if resolution != (1, 1, 1):
                    # divide all three at once
                    nodes_df[['x', 'y', 'z']] = (
                        nodes_df[['x', 'y', 'z']]
                        .div(self.resolution, axis=1)  # resolution is a (3,) tuple
                    )

                # If PYMAID is available, create CatmaidNeuron objects
                if PYMAID_AVAILABLE:
                    neuron = pymaid.CatmaidNeuron(nodes_df)
                    neuron.skeleton_id = str(skeleton_id)
                    neurons.append(neuron)
                else:
                    # Create a simple dictionary structure if pymaid is not available
                    neuron = {
                        'skeleton_id': str(skeleton_id),
                        'nodes': nodes_df
                    }
                    neurons.append(neuron)

                # store raw nodes for plotting and merging
                sid = str(skeleton_id)
                self.skeleton_nodes_map[sid] = nodes_df

            if PYMAID_AVAILABLE:
                self.traced_neurons = pymaid.CatmaidNeuronList(neurons)
            else:
                self.traced_neurons = neurons

            print(f"Loaded {len(neurons)} neurons from CSV file")

        except Exception as e:
            raise Exception(f"Failed to load skeletons from CSV: {e}")

    def load_precomputed_segmentation(self) -> None:
        """
        Load the precomputed segmentation from the Google Cloud bucket.
        """
        if not CLOUDVOLUME_AVAILABLE:
            raise ImportError("cloud-volume is required to load precomputed segmentation")

        if not self.cv:
            raise ValueError("No Cloud Volume connection established. Provide gcloud_bucket first.")

        print("Precomputed segmentation is accessible through the CloudVolume instance.")
        print(f"Segmentation info: {self.cv.info}")

    def plot_traced_neurons(self, figsize=(10, 10), color='blue', show=False) -> plt.Figure:
        """
        Plot the traced neuron arbors in 3D.
        
        Args:
            figsize: Figure size as tuple (width, height)
            color: Color for the neuron traces
            show: Whether to show the plot immediately
            
        Returns:
            matplotlib Figure object
        """
        if self.traced_neurons is None:
            raise ValueError("No traced neurons loaded. Load skeletons first.")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if PYMAID_AVAILABLE and isinstance(self.traced_neurons, pymaid.CatmaidNeuronList):
            # Plot using pymaid functionality
            for neuron in self.traced_neurons:
                nodes = neuron.nodes
                for i, row in nodes.iterrows():
                    if row['parent_id'] != -1:  # Skip root nodes
                        # Find parent node
                        parent = nodes[nodes['node_id'] == row['parent_id']]
                        if not parent.empty:
                            # Draw line from parent to current node
                            ax.plot([parent['x'].values[0], row['x']],
                                    [parent['y'].values[0], row['y']],
                                    [parent['z'].values[0], row['z']],
                                    color=color, linewidth=1)
        else:
            # Plot using the simple dictionary structure
            for neuron in self.traced_neurons:
                nodes = neuron['nodes']
                for i, row in nodes.iterrows():
                    if row['parent_id'] != -1:  # Skip root nodes
                        # Find parent node
                        parent = nodes[nodes['node_id'] == row['parent_id']]
                        if not parent.empty:
                            # Draw line from parent to current node
                            ax.plot([parent['x'].values[0], row['x']],
                                    [parent['y'].values[0], row['y']],
                                    [parent['z'].values[0], row['z']],
                                    color=color, linewidth=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Traced Neuron Arbors')

        if show:
            plt.show()

        return fig

    def check_segmentation_ids(self, padding: int = 0) -> Dict:
        """
        Check all node locations from the tracing and compare segment IDs from the segmentation.
        Maintains a list for merging segmentation IDs which will lead to the tracing in CATMAID.
        
        Args:
            padding: Number of voxels to pad around each node when checking segmentation
            
        Returns:
            Dictionary mapping skeleton IDs to lists of segment IDs
        """
        if not CLOUDVOLUME_AVAILABLE:
            raise ImportError("cloud-volume is required to check segmentation IDs")

        if self.traced_neurons is None:
            raise ValueError("No traced neurons loaded. Load skeletons first.")

        if self.cv is None:
            raise ValueError("No Cloud Volume connection established. Provide gcloud_bucket first.")

        # Reset segment IDs map
        self.segment_ids_map = {}

        # Process each neuron
        if PYMAID_AVAILABLE and isinstance(self.traced_neurons, pymaid.CatmaidNeuronList):
            for neuron in self.traced_neurons:
                skeleton_id = neuron.skeleton_id
                if DEBUG:
                    neuron.nodes = neuron.nodes[:DEBUG_N]
                self.segment_ids_map[skeleton_id] = self._process_neuron_nodes(neuron.nodes, padding)
        else:
            for neuron in self.traced_neurons:
                skeleton_id = neuron['skeleton_id']
                if DEBUG:
                    neuron.nodes = neuron.nodes[:DEBUG_N]
                self.segment_ids_map[skeleton_id] = self._process_neuron_nodes(neuron['nodes'], padding)

        print(f"segment_ids_map: {self.segment_ids_map}")
        # Find merge candidates
        self._find_merge_candidates()

        return self.segment_ids_map

    def _process_neuron_nodes(self, nodes_df: pd.DataFrame, padding: int) -> List[int]:
        """
        Process nodes of a neuron to get segment IDs from the segmentation.
        
        Args:
            nodes_df: DataFrame with node coordinates
            padding: Number of voxels to pad around each node
            
        Returns:
            List of unique segment IDs
        """
        segment_ids = []

        for _, node in tqdm(nodes_df.iterrows(), desc="iterating over nodes of a skid", total=len(nodes_df)):
            # Convert node coordinates to voxel coordinates in the segmentation
            x, y, z = int(node['x']), int(node['y']), int(node['z'])

            # Get segmentation ID at this point
            try:
                # If padding is 0, just get the single voxel
                if padding == 0:
                    seg_id = self.cv[x:x + 1, y:y + 1, z:z + 1][0, 0, 0]
                    if seg_id != 0:  # Ignore background
                        segment_ids.append(int(seg_id))
                else:
                    # Get a cube around the point
                    cube = self.cv[x - padding:x + padding + 1,
                           y - padding:y + padding + 1,
                           z - padding:z + padding + 1]
                    # Get unique IDs in the cube (excluding background)
                    unique_ids = np.unique(cube)
                    unique_ids = unique_ids[unique_ids != 0]  # Remove background
                    segment_ids.extend([int(id) for id in unique_ids])
            except Exception as e:
                print(f"Error getting segmentation at ({x}, {y}, {z}): {e}")

        # Return unique segment IDs
        return list(set(segment_ids))

    def _find_merge_candidates(self) -> None:
        """
        Find candidates for merging based on segment IDs that appear in multiple neurons.
        """
        # Reset merge candidates
        self.merge_candidates = []

        # Create a mapping from segment ID to skeleton IDs
        segment_to_skeletons = {}

        for skeleton_id, segment_ids in self.segment_ids_map.items():
            for segment_id in segment_ids:
                if segment_id not in segment_to_skeletons:
                    segment_to_skeletons[segment_id] = []
                segment_to_skeletons[segment_id].append(skeleton_id)

        # find all
        self.merge_candidates = [
            {'skeleton_id': sk, 'segment_ids': segs}
            for sk, segs in self.segment_ids_map.items()
        ]

        print(f"Prepared {len(self.merge_candidates)} skeleton-level merge groups")

    def plot_merge_candidates(self, max_candidates: int = 5, figsize=(15, 10), show=False) -> plt.Figure:
        """
        Plot the putatively merged segments.
        
        Args:
            max_candidates: Maximum number of merge candidates to plot
            figsize: Figure size as tuple (width, height)
            show: Whether to show the plot immediately
            
        Returns:
            matplotlib Figure object
        """
        if not CLOUDVOLUME_AVAILABLE:
            raise ImportError("cloud-volume is required to plot merge candidates")

        if self.cv is None:
            raise ValueError("No Cloud Volume connection established. Provide gcloud_bucket first.")

        if not self.merge_candidates:
            raise ValueError("No merge candidates found. Run check_segmentation_ids first.")

        # Limit the number of candidates to plot
        candidates_to_plot = self.merge_candidates[:min(max_candidates, len(self.merge_candidates))]

        # Create figure
        fig = plt.figure(figsize=figsize)

        for i, candidate in enumerate(candidates_to_plot):
            segment_id = candidate['segment_id']
            skeleton_ids = candidate['skeleton_ids']

            # Get the segment from the segmentation
            try:
                # This is a simplified approach - in a real scenario, you would need to
                # determine the bounding box of the segment more efficiently
                # For demonstration purposes, we'll just create a small cube around a point

                # Find a node that maps to this segment ID
                point = None
                for skeleton_id in skeleton_ids:
                    if PYMAID_AVAILABLE and isinstance(self.traced_neurons, pymaid.CatmaidNeuronList):
                        neuron = self.traced_neurons[self.traced_neurons.skeleton_id == skeleton_id][0]
                        nodes = neuron.nodes
                    else:
                        neuron = next(n for n in self.traced_neurons if n['skeleton_id'] == skeleton_id)
                        nodes = neuron['nodes']

                    # Sample a point from the nodes
                    point = (int(nodes['x'].iloc[0]), int(nodes['y'].iloc[0]), int(nodes['z'].iloc[0]))
                    break

                if point:
                    x, y, z = point
                    # Get a cube around the point
                    size = 50  # Size of the cube
                    cube = self.cv[x - size:x + size, y - size:y + size, z - size:z + size]

                    # Create a binary mask for this segment
                    mask = (cube == segment_id)

                    # Plot the segment
                    ax = fig.add_subplot(1, len(candidates_to_plot), i + 1, projection='3d')
                    ax.voxels(mask, facecolors='red', alpha=0.5)
                    ax.set_title(f"Segment {segment_id}\nSkeletons: {', '.join(skeleton_ids)}")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
            except Exception as e:
                print(f"Error plotting segment {segment_id}: {e}")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def calculate_metrics(self, ground_truth_segmentation=None) -> Dict:
        """
        Calculate metrics to evaluate the quality of recovery.
        
        Args:
            ground_truth_segmentation: Optional ground truth segmentation for comparison
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}

        # If we have ground truth segmentation, calculate NERL and other metrics
        if ground_truth_segmentation is not None and GRAPHTOOL_AVAILABLE:
            try:
                # Calculate NERL (Normalized Expected Run Length)
                nerl = funlib.evaluate.expected_run_length(
                    ground_truth_segmentation,
                    self.cv[:],  # Get the full segmentation
                    normalize=True
                )
                metrics['nerl'] = nerl

                # Calculate VOI (Variation of Information)
                voi_split, voi_merge = funlib.evaluate.voi(
                    ground_truth_segmentation,
                    self.cv[:]
                )
                metrics['voi_split'] = voi_split
                metrics['voi_merge'] = voi_merge

                # Calculate Rand index
                rand = funlib.evaluate.rand(
                    ground_truth_segmentation,
                    self.cv[:]
                )
                metrics['rand'] = rand

            except Exception as e:
                print(f"Error calculating metrics: {e}")
        else:
            # Calculate basic metrics without ground truth
            if self.segment_ids_map:
                # Count total segments per neuron
                segments_per_neuron = {k: len(v) for k, v in self.segment_ids_map.items()}
                metrics['segments_per_neuron'] = segments_per_neuron

                # Count total merge candidates
                metrics['merge_candidates_count'] = len(self.merge_candidates)

                # Calculate fragmentation index (higher means more fragmented)
                total_neurons = len(self.segment_ids_map)
                total_segments = sum(len(v) for v in self.segment_ids_map.values())
                if total_neurons > 0:
                    metrics['fragmentation_index'] = total_segments / total_neurons
                else:
                    metrics['fragmentation_index'] = 0

        return metrics

    def merge_segments(self, segment_ids: List[int], cave_client=None) -> bool:
        """
        Merge segments using caveclient.
        
        Args:
            segment_ids: List of segment IDs to merge
            cave_client: Optional caveclient instance
            
        Returns:
            Boolean indicating success
        """
        if not CAVECLIENT_AVAILABLE:
            raise ImportError("caveclient is required to merge segments")

        if cave_client is None and not hasattr(self, 'cave_client'):
            raise ValueError("No caveclient provided. Pass a caveclient instance.")

        client = cave_client if cave_client is not None else self.cave_client

        try:
            # This is a placeholder for the actual merge operation
            # The actual implementation would depend on the specific API of caveclient
            # and the chunked graph structure

            # Example (pseudocode):
            # client.chunkedgraph.merge_edges(segment_ids)

            print(f"Merging segments: {segment_ids}")
            print("Note: This is a placeholder. Implement the actual merge operation based on caveclient API.")

            return True
        except Exception as e:
            print(f"Error merging segments: {e}")
            return False

    def merge_skeleton_fragments_local(self, skeleton_id: Union[int, str]) -> None:
        """
        “Merge” all fragment segment IDs of a given skeleton into one master ID,
        choosing as master the fragment that covers the most traced nodes.
        """
        sid = str(skeleton_id)
        segs = self.segment_ids_map.get(sid, [])
        if len(segs) < 2:
            print(f"Skeleton {sid} has {len(segs)} fragment(s); nothing to merge.")
            return

        # 1) Recompute seg_id at each node (as in plot_merge_comparison)
        nodes = self.skeleton_nodes_map[sid].copy()
        # res   = self.resolution
        coords = nodes[['x','y','z']].astype(float)
        # if res != (1,1,1):
        #     coords = coords.div(res, axis=1)
        pix = coords.round().astype(int)
        seg_at_node = []
        for x, y, z in pix.values:
            block = self.cv[x:x+1, y:y+1, z:z+1]
            # block.flat[0] is the first element as a scalar (ndarray → Python scalar)
            val = block.flat[0]
            seg_at_node.append(int(val))
        nodes['seg_id'] = seg_at_node

        # 2) Count how many nodes each fragment contributes
        counts = nodes['seg_id'].value_counts().to_dict()
        # Filter to only your candidate segs
        counts = {seg: counts.get(seg, 0) for seg in segs}

        # 3) Pick the fragment with the highest node‐count as master
        master, max_count = max(counts.items(), key=lambda kv: kv[1])
        print(f"Chosen master={master} with {max_count} of {len(nodes)} total nodes")

        # 4) Build local remap old→master
        self._local_remap = getattr(self, '_local_remap', {})
        for old in segs:
            self._local_remap[old] = master

        # 5) Collapse the listing in segment_ids_map
        self.segment_ids_map[sid] = [master]
        print(f"Skeleton {sid}: merged {len(segs)} → 1 segment (master={master})")

    def merge_skeleton_fragments_cave(self,
                                      skeleton_id: Union[int, str],
                                      cave_client=None) -> bool:
        """
        Merge _all_ segment fragments belonging to one skeleton into a single segment ID.

        Returns True if a merge was attempted (and caveclient returned success),
        False if there was nothing to merge.
        """
        # 1) Look up the list of segment-IDs for this skeleton
        segs = self.segment_ids_map.get(str(skeleton_id), [])
        if len(segs) < 2:
            print(f"Skeleton {skeleton_id} has {len(segs)} fragment(s); nothing to merge.")
            return False

        # 2) Choose a master ID (here: the max; you could also choose by voxel count)
        master = max(segs)
        to_merge = segs[:]  # including the master itself

        # 3) Run your caveclient-based merge
        success = self.merge_segments(to_merge, cave_client=cave_client)
        if success:
            # 4) Update in-memory maps so that this skeleton now only points at the master
            self.segment_ids_map[str(skeleton_id)] = [master]
            # Remove any stale “merge candidate” entries
            self.merge_candidates = [
                m for m in self.merge_candidates
                if str(m.get('skeleton_id')) != str(skeleton_id)
            ]
            print(f"Skeleton {skeleton_id}: merged {len(segs)} → 1 segment ID ({master})")
        else:
            print(f"Merge failed for skeleton {skeleton_id}")

        return success

    def plot_fragments(self,
                       skeleton_id: Union[int, str],
                       figsize=(8, 6),
                       show: bool = True) -> plt.Figure:
        """
        Scatter-plot each node of a given skeleton, color-coded by its current segment ID.

        Node coordinates are adjusted by the resolution provided at load time.
        """

        sid = str(skeleton_id)
        segs = self.segment_ids_map.get(sid, [])
        if not segs:
            raise ValueError(f"No segments found for skeleton {skeleton_id}")

        # Grab the traced-neuron’s node DataFrame
        if PYMAID_AVAILABLE and isinstance(self.traced_neurons, pymaid.CatmaidNeuronList):
            neuron = next(n for n in self.traced_neurons if n.skeleton_id == sid)
            nodes = neuron.nodes.copy()
        else:
            neuron = next(n for n in self.traced_neurons if n['skeleton_id'] == sid)
            nodes = neuron['nodes'].copy()

        if DEBUG:
            nodes = nodes[:DEBUG_N]
        # Compute pixel coordinates by dividing by resolution
        # self.resolution should have been set during load_skeletons_from_csv
        res = getattr(self, 'resolution', (1, 1, 1))
        coords = nodes[['x', 'y', 'z']].astype(float)
        if res != (1, 1, 1):
            coords = coords.div(res, axis=1)
        # Round and convert to int for indexing
        pix = coords.round().astype(int)

        # Tag each node with its segmentation ID
        seg_at_node = []
        for x, y, z in pix.values:
            block = self.cv[x:x+1, y:y+1, z:z+1]
            # block.flat[0] is the first element as a scalar (ndarray → Python scalar)
            val = block.flat[0]
            seg_at_node.append(int(val))
        nodes['seg_id'] = seg_at_node

        # Plot fragments
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        cmap = plt.cm.get_cmap('tab10', len(segs))

        for i, segment_id in enumerate(segs):
            sub_idx = [idx for idx, sid_val in enumerate(nodes['seg_id']) if sid_val == segment_id]
            if not sub_idx:
                continue
            sub = coords.iloc[sub_idx]
            ax.scatter(sub['x'], sub['y'], sub['z'],
                       s=5,
                       label=str(segment_id),
                       color=cmap(i))

        ax.set_title(f"Skeleton {skeleton_id} fragments (n={len(segs)})")
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Z (px)')
        ax.legend(loc='best')

        if show:
            plt.show()
        return fig

    def plot_merge_comparison(self,
                              skeleton_id: Union[int, str],
                              figsize=(14, 6),
                              point_size: int = 30,
                              show: bool = True,
                              savefig_name="plot_before_after.png") -> plt.Figure:
        """
        Side‐by‐side 3D view of one skeleton:
          – LEFT: drawn as gray lines, with nodes overlaid colored by fragment ID
          – RIGHT: same skeleton, overlaid as one color (master segment)
        """
        sid = str(skeleton_id)
        # 1) get the loaded nodes & their pixel coords
        nodes = self.skeleton_nodes_map[sid].copy()
        res = self.resolution
        coords = nodes[['x','y','z']].astype(float)
        # if res != (1,1,1):
        #     coords = coords.div(res, axis=1)

        # 2) lookup seg_id for each node
        pix = coords.round().astype(int)
        seg_at_node = []
        for x, y, z in pix.values:
            block = self.cv[x:x+1, y:y+1, z:z+1]
            # block.flat[0] is the first element as a scalar (ndarray → Python scalar)
            val = block.flat[0]
            seg_at_node.append(int(val))
        nodes['seg_id'] = seg_at_node

        print(f"nodes at plot: {nodes}")
        # save the nodes with seg_ids file per skid
        nodes.to_csv(f"./nodes_w_segids_{sid}.csv", index=False)

        # 3) apply your local remap for after‐merge
        frags = self.segment_ids_map[sid]
        if hasattr(self, '_local_remap'):
            nodes['seg_id_mapped'] = nodes['seg_id'].map(lambda old: self._local_remap.get(old, old))
        else:
            nodes['seg_id_mapped'] = nodes['seg_id']
        master = nodes['seg_id_mapped'].iloc[0]

        # 4) prepare figure + axes
        fig = plt.figure(figsize=figsize)
        ax_pre  = fig.add_subplot(1, 2, 1, projection='3d')
        ax_post = fig.add_subplot(1, 2, 2, projection='3d')

        # helper to draw the skeleton lines
        def draw_skeleton(ax):
            for _, row in nodes.iterrows():
                pid = row['parent_id']
                if pd.notna(pid):
                    # find parent coords
                    p = nodes[nodes['node_id']==pid]
                    if not p.empty:
                        x0,y0,z0 = row[['x','y','z']].values
                        x1,y1,z1 = p[['x','y','z']].values[0]
                        ax.plot([x0, x1], [y0, y1], [z0, z1],
                                color='gray', alpha=0.5, linewidth=1)

        # draw the base skeleton in both panes
        draw_skeleton(ax_pre)
        draw_skeleton(ax_post)

        # 5) scatter points for pre-merge
        cmap = plt.cm.get_cmap('tab10', len(frags))
        for i, seg in enumerate(frags):
            mask = nodes['seg_id']==seg
            pts = coords[mask]
            ax_pre.scatter(pts['x'], pts['y'], pts['z'],
                           s=point_size,
                           color=cmap(i),
                           label=str(seg),
                           edgecolor='k', linewidth=0.5)

        ax_pre.set_title(f"Skeleton {sid} — Before Merge")
        ax_pre.legend(loc='best', fontsize='small')

        # 6) scatter points for post-merge
        mask2 = nodes['seg_id_mapped']==master
        pts2 = coords[mask2]
        ax_post.scatter(pts2['x'], pts2['y'], pts2['z'],
                        s=point_size,
                        color='orange',
                        label=f"master {master}",
                        edgecolor='k', linewidth=0.5)
        ax_post.set_title(f"Skeleton {sid} — After Merge")
        ax_post.legend(loc='best')

        # 7) common formatting
        for ax in (ax_pre, ax_post):
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            ax.set_zlabel('Z (px)')
            ax.view_init(elev=20, azim=45)  # consistent view

        plt.tight_layout()
        if show:
            plt.show()

        fig.savefig(f"./{savefig_name}", dpi=300)
        return fig


def parse_int_list(s: str) -> list[int]:
    """
    Convert a comma-separated string to a list of ints.
    e.g. "1,2,3" → [1, 2, 3]
    """
    try:
        items = [int(x) for x in s.split(',') if x]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer in list: {s!r}")
    return items


def parse_resolution(s: str) -> tuple[int, int, int]:
    """
    Convert a comma-separated string to a 3-tuple of ints.
    e.g. "8,8,8" → (8, 8, 8)
    """
    try:
        parts = [int(x) for x in s.split(',') if x]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer in resolution: {s!r}")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Resolution must have exactly three values (got {len(parts)}): {s!r}"
        )
    return tuple(parts)


def main():
    """
    Main function to demonstrate usage of the NeuronRecovery class.
    """
    parser = argparse.ArgumentParser(description='Neuron Segmentation Recovery Tool')
    parser.add_argument('--gcloud-bucket', default="gs://zlatic-lab/octo_8x8x8_full_200525/fragments",
                        type=str, help='Google Cloud bucket URL with precomputed segmentation')
    parser.add_argument('--csv-file',
                        default="/media/samia/DATA/mounts/fibserver1/smohinta_data/proofreading/tracedCATMAID_neuroglancer/octo_229050_nodes.csv",
                        type=str, help='CSV file with traced skeleton nodes')
    parser.add_argument('--catmaid-server', type=str, default="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/"
                        , help='CATMAID server URL')
    parser.add_argument('--catmaid-api-token', default="90f00590555eb256f256d1062e38521cbe180293", type=str,
                        help='CATMAID API token')
    parser.add_argument('--catmaid-http-user', default="smohinta", type=str, help='CATMAID HTTP user')
    parser.add_argument('--catmaid-http-password', default="headset-recovery-handshake", type=str,
                        help='CATMAID HTTP password')
    parser.add_argument('--skeleton-ids', default=[229050], type=parse_int_list,
                        help='Comma-separated list of skeleton IDs to load from CATMAID')
    parser.add_argument('--resolution', default="8,8,8", type=parse_resolution,
                        help='Comma-separated list of skeleton IDs to load from CATMAID')

    args = parser.parse_args()

    # check data resolution passed
    resolution = args.resolution

    # Initialize NeuronRecovery
    catmaid_credentials = None
    if args.catmaid_server:
        catmaid_credentials = {
            'server': args.catmaid_server,
            'api_token': args.catmaid_api_token,
            'http_user': args.catmaid_http_user,
            'http_password': args.catmaid_http_password
        }

    recovery = NeuronRecovery(
        gcloud_bucket=args.gcloud_bucket,
        catmaid_credentials=catmaid_credentials
    )

    # Load skeletons
    if args.csv_file:
        recovery.load_skeletons_from_csv(args.csv_file, resolution=resolution)
    elif args.skeleton_ids:
        skeleton_ids = [int(skid.strip()) for skid in args.skeleton_ids.split(',')]
        recovery.load_skeletons_from_catmaid(skeleton_ids)  # check later how to navigate resolution
    else:
        print("No skeleton data provided. Use --csv-file or --skeleton-ids.")
        return

    # Load precomputed segmentation
    if args.gcloud_bucket:
        recovery.load_precomputed_segmentation()
    else:
        print("No Google Cloud bucket provided. Use --gcloud-bucket.")
        return

    # Plot traced neurons
    recovery.plot_traced_neurons(show=False)

    # Check segmentation IDs
    segment_ids_map = recovery.check_segmentation_ids(padding=0)
    print("Segment IDs per neuron:")
    for skeleton_id, segment_ids in segment_ids_map.items():
        print(f"Skeleton {skeleton_id}: {len(segment_ids)} segments")

    # Once you’ve computed segment_ids_map...
    for sk in segment_ids_map:
        print(f"\nComparing pre- and post-merge for skeleton {sk}…")
        # Call the comparison plot (this will show both panels)
        recovery.plot_merge_comparison(sk, savefig_name="plot_before_merge.png")

        # Now actually perform the in-memory merge
        print(f"Merging fragments for skeleton {sk}…")
        recovery.merge_skeleton_fragments_local(sk)

        # And re-plot (the right panel will now show all nodes recolored to master)
        print(f"Redrawing after merge for skeleton {sk}…")
        recovery.plot_merge_comparison(sk, savefig_name="plot_after_merge.png")


    # # Plot merge candidates
    # if recovery.merge_candidates:
    #     print(f"merge candidates: {recovery.merge_candidates}")
    #     recovery.plot_merge_candidates()

    # # Visualize fragments before merge
    # for sk in segment_ids_map:
    #     print(f"Plotting fragments for skeleton {sk}...")
    #     recovery.plot_fragments(sk)
    #
    # # Merge all fragments per skeleton
    # for sk in list(segment_ids_map.keys()):
    #     print(f"Merging fragments for skeleton {sk}...")
    #     recovery.merge_skeleton_fragments_local(sk)
    #
    # # Visualize merged skeletons
    # for sk in segment_ids_map:
    #     print(f"Plotting merged skeleton {sk}...")
    #     recovery.plot_fragments(sk)

    # # Calculate metrics
    # metrics = recovery.calculate_metrics()
    # print("\nMetrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
