import sys
import os
import numpy as np
import zarr
import tifffile as tff
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from typing import Tuple
from collections import defaultdict


#
# def remove_small_instances(segm, thres_small, mode='background'):
#     """Remove small instances from segmentation."""
#     if mode == 'background':
#         return remove_small_objects(segm, min_size=thres_small, connectivity=1)
#     else:
#         # For 'neighbor' mode, replace small objects with nearest neighbor
#         cleaned = remove_small_objects(segm, min_size=thres_small, connectivity=1)
#         # This is a simplified version - full implementation would need nearest neighbor assignment
#         return cleaned


def remove_small_instances(segm, thres_small=128, mode='background'):
    """Remove small spurious instances.
    """
    assert mode in ['background', 'neighbor']

    if mode == 'background':
        return remove_small_objects(segm, thres_small)

    seg_idx = np.unique(segm)[1:]
    for idx in seg_idx:
        temp = (segm == idx).astype(np.uint8)
        if temp.sum() < thres_small:
            temp_dilated = dilation(temp, np.ones((1, 3, 3)))
            diff = temp_dilated - temp
            diff_mask = segm.copy()
            diff_mask[np.where(diff == 0)] = 0
            touch_idx, counts = np.unique(diff_mask, return_counts=True)

            if len(touch_idx) > 1 and touch_idx[0] == 0:
                touch_idx = touch_idx[1:]
                counts = counts[1:]

            segm[np.where(segm == idx)] = touch_idx[np.argmax(counts)]

    return segm


def binary_connected(volume_chunk_01, thres_binary=0.5, thres_small=128,
                     scale_factors=(1.0, 1.0, 1.0), remove_small_mode='background'):
    """Convert binary foreground probability map to instance masks via connected components."""
    if volume_chunk_01.ndim != 3:
        raise ValueError(f"Input volume_chunk_01 must be 3D. Got shape {volume_chunk_01.shape}")

    # Binarize if input is probabilities
    foreground = (volume_chunk_01 > thres_binary).astype(np.uint8)

    # Connected component labeling
    segm = label(foreground, connectivity=1)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    # Apply scaling if needed
    if not all(x == 1.0 for x in scale_factors):
        target_size = (int(foreground.shape[0] * scale_factors[0]),
                       int(foreground.shape[1] * scale_factors[1]),
                       int(foreground.shape[2] * scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return segm.astype(np.uint32)


class GlobalLabelManager:
    """Manages consistent global labeling across chunks."""

    def __init__(self):
        self.next_global_id = 1
        self.local_to_global = {}  # Maps (chunk_id, local_label) -> global_label
        self.global_equivalences = {}  # Maps global_label -> canonical_global_label

    def get_canonical_label(self, global_label):
        """Get the canonical (root) label for a global label."""
        if global_label not in self.global_equivalences:
            return global_label

        # Path compression
        root = global_label
        while root in self.global_equivalences:
            root = self.global_equivalences[root]

        # Update all nodes in path to point directly to root
        current = global_label
        while current in self.global_equivalences and self.global_equivalences[current] != root:
            next_node = self.global_equivalences[current]
            self.global_equivalences[current] = root
            current = next_node

        return root

    def merge_labels(self, global_label1, global_label2):
        """Merge two global labels."""
        root1 = self.get_canonical_label(global_label1)
        root2 = self.get_canonical_label(global_label2)

        if root1 != root2:
            # Merge smaller ID into larger ID for consistency
            if root1 < root2:
                self.global_equivalences[root2] = root1
            else:
                self.global_equivalences[root1] = root2

    def assign_global_labels(self, chunk_id, local_labels):
        """Assign global labels to local labels in a chunk."""
        global_labels = {}

        for local_label in local_labels:
            if local_label == 0:  # Background
                global_labels[local_label] = 0
                continue

            key = (chunk_id, local_label)
            if key not in self.local_to_global:
                self.local_to_global[key] = self.next_global_id
                self.next_global_id += 1

            global_labels[local_label] = self.local_to_global[key]

        return global_labels


def get_overlap_regions(chunk_shape, overlap):
    """Define overlap regions for boundary matching."""
    z_size, y_size, x_size = chunk_shape
    oz, oy, ox = overlap

    regions = {}

    # Face regions for matching with adjacent chunks
    if oz > 0:
        regions['z_front'] = (slice(0, oz), slice(None), slice(None))
        regions['z_back'] = (slice(z_size - oz, z_size), slice(None), slice(None))

    if oy > 0:
        regions['y_front'] = (slice(None), slice(0, oy), slice(None))
        regions['y_back'] = (slice(None), slice(y_size - oy, y_size), slice(None))

    if ox > 0:
        regions['x_front'] = (slice(None), slice(None), slice(0, ox))
        regions['x_back'] = (slice(None), slice(None), slice(x_size - ox, x_size))

    return regions


def match_boundary_labels(current_labels, previous_labels, threshold=0.5):
    """Match labels across chunk boundaries based on overlap."""
    matches = []

    # Get unique labels (excluding background)
    current_unique = np.unique(current_labels[current_labels > 0])
    previous_unique = np.unique(previous_labels[previous_labels > 0])

    if len(current_unique) == 0 or len(previous_unique) == 0:
        return matches

    # For each current label, find best matching previous label
    for curr_label in current_unique:
        curr_mask = (current_labels == curr_label)

        best_prev_label = 0
        best_overlap_ratio = 0

        for prev_label in previous_unique:
            prev_mask = (previous_labels == prev_label)

            # Calculate overlap
            intersection = np.sum(curr_mask & prev_mask)
            union = np.sum(curr_mask | prev_mask)

            if union > 0:
                overlap_ratio = intersection / union
                if overlap_ratio > best_overlap_ratio and overlap_ratio > threshold:
                    best_overlap_ratio = overlap_ratio
                    best_prev_label = prev_label

        if best_prev_label > 0:
            matches.append((curr_label, best_prev_label))

    return matches


def convert_semantic_to_instances_chunked(
        input_file_path: str,
        output_dir: str,
        output_format: str = "zarr",
        chunk_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: Tuple[int, int, int] = (16, 16, 16),
        method: str = "binary_connected",
        thres_foreground: float = 0.5,
        thres_small_instances: int = 128,
        scale_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        remove_small_mode: str = 'background'
):
    """
    Convert semantic segmentation to instance masks with proper chunked processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename_base = os.path.splitext(os.path.basename(input_file_path))[0]
    output_filepath = os.path.join(output_dir, f"{output_filename_base}_instances.{output_format}")

    print(f"Loading semantic prediction from: {input_file_path}")

    # Load input data
    if input_file_path.endswith('.zarr'):
        input_data = zarr.open(input_file_path, mode='r')
        volume_shape = input_data.shape
    elif input_file_path.endswith(('.tif', '.tiff')):
        # For large TIFF files, use memory mapping
        input_data = tff.memmap(input_file_path, mode='r')
        volume_shape = input_data.shape
    else:
        raise ValueError(f"Unsupported input format: {input_file_path}")

    if method != "binary_connected":
        raise NotImplementedError(f"Method '{method}' not supported.")

    print(f"Volume shape: {volume_shape}")
    print(f"Output will be saved to: {output_filepath}")
    print(f"Processing with chunk size: {chunk_size}, overlap: {overlap}")

    # Initialize output
    if output_format == "zarr":
        output_data = zarr.open(
            output_filepath,
            mode='w',
            shape=volume_shape,
            dtype=np.uint32,
            chunks=chunk_size,
            compressor=zarr.GZip()
        )
    else:
        print("Warning: TIFF output requires loading full volume into memory.")
        output_data = np.zeros(volume_shape, dtype=np.uint32)

    # Initialize global label manager
    label_manager = GlobalLabelManager()

    # Calculate chunk positions
    step_z = max(1, chunk_size[0] - overlap[0])
    step_y = max(1, chunk_size[1] - overlap[1])
    step_x = max(1, chunk_size[2] - overlap[2])

    z_positions = list(range(0, volume_shape[0], step_z))
    y_positions = list(range(0, volume_shape[1], step_y))
    x_positions = list(range(0, volume_shape[2], step_x))

    # Adjust last positions to cover entire volume
    if z_positions[-1] + chunk_size[0] < volume_shape[0]:
        z_positions.append(max(0, volume_shape[0] - chunk_size[0]))
    if y_positions[-1] + chunk_size[1] < volume_shape[1]:
        y_positions.append(max(0, volume_shape[1] - chunk_size[1]))
    if x_positions[-1] + chunk_size[2] < volume_shape[2]:
        x_positions.append(max(0, volume_shape[2] - chunk_size[2]))

    total_chunks = len(z_positions) * len(y_positions) * len(x_positions)
    print(f"Processing {total_chunks} chunks...")

    # Store boundary data for matching
    boundary_data = {}

    chunk_count = 0
    for z_idx, z_start in enumerate(z_positions):
        for y_idx, y_start in enumerate(y_positions):
            for x_idx, x_start in enumerate(x_positions):

                # Calculate chunk bounds
                z_end = min(z_start + chunk_size[0], volume_shape[0])
                y_end = min(y_start + chunk_size[1], volume_shape[1])
                x_end = min(x_start + chunk_size[2], volume_shape[2])

                # Extract chunk
                chunk_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
                semantic_chunk = np.array(input_data[chunk_slice])

                # Process chunk to get local instance labels
                local_instances = binary_connected(
                    semantic_chunk,
                    thres_binary=thres_foreground,
                    thres_small=thres_small_instances,
                    scale_factors=scale_factors,
                    remove_small_mode=remove_small_mode
                )

                # Get unique local labels
                local_labels = np.unique(local_instances)
                local_labels = local_labels[local_labels > 0]  # Exclude background

                if len(local_labels) == 0:
                    # No instances in this chunk
                    output_data[chunk_slice] = 0
                    chunk_count += 1
                    continue

                # Assign initial global labels
                chunk_id = (z_idx, y_idx, x_idx)
                global_label_map = label_manager.assign_global_labels(chunk_id, local_labels)

                # Check boundaries with previous chunks and merge labels
                overlap_regions = get_overlap_regions(local_instances.shape, overlap)

                # Check Z boundary (previous z-chunk)
                if z_idx > 0 and 'z_front' in overlap_regions:
                    prev_chunk_id = (z_idx - 1, y_idx, x_idx)
                    if prev_chunk_id in boundary_data and 'z_back' in boundary_data[prev_chunk_id]:
                        current_boundary = local_instances[overlap_regions['z_front']]
                        previous_boundary = boundary_data[prev_chunk_id]['z_back']

                        matches = match_boundary_labels(current_boundary, previous_boundary)
                        for curr_local, prev_local in matches:
                            if curr_local in global_label_map and prev_local in \
                                    boundary_data[prev_chunk_id]['boundary_global_map']:
                                curr_global = global_label_map[curr_local]
                                prev_global = boundary_data[prev_chunk_id]['boundary_global_map'][prev_local]
                                label_manager.merge_labels(curr_global, prev_global)

                # Check Y boundary (previous y-chunk)
                if y_idx > 0 and 'y_front' in overlap_regions:
                    prev_chunk_id = (z_idx, y_idx - 1, x_idx)
                    if prev_chunk_id in boundary_data and 'y_back' in boundary_data[prev_chunk_id]:
                        current_boundary = local_instances[overlap_regions['y_front']]
                        previous_boundary = boundary_data[prev_chunk_id]['y_back']

                        matches = match_boundary_labels(current_boundary, previous_boundary)
                        for curr_local, prev_local in matches:
                            if curr_local in global_label_map and prev_local \
                                    in boundary_data[prev_chunk_id]['boundary_global_map']:
                                curr_global = global_label_map[curr_local]
                                prev_global = boundary_data[prev_chunk_id]['boundary_global_map'][prev_local]
                                label_manager.merge_labels(curr_global, prev_global)

                # Check X boundary (previous x-chunk)
                if x_idx > 0 and 'x_front' in overlap_regions:
                    prev_chunk_id = (z_idx, y_idx, x_idx - 1)
                    if prev_chunk_id in boundary_data and 'x_back' in boundary_data[prev_chunk_id]:
                        current_boundary = local_instances[overlap_regions['x_front']]
                        previous_boundary = boundary_data[prev_chunk_id]['x_back']

                        matches = match_boundary_labels(current_boundary, previous_boundary)
                        for curr_local, prev_local in matches:
                            if curr_local in global_label_map and prev_local in boundary_data[prev_chunk_id][
                                'boundary_global_map']:
                                curr_global = global_label_map[curr_local]
                                prev_global = boundary_data[prev_chunk_id]['boundary_global_map'][prev_local]
                                label_manager.merge_labels(curr_global, prev_global)

                # Apply final global labels with canonical mapping
                final_instances = np.zeros_like(local_instances)
                for local_label in local_labels:
                    global_label = global_label_map[local_label]
                    canonical_label = label_manager.get_canonical_label(global_label)
                    final_instances[local_instances == local_label] = canonical_label

                # Write to output
                output_data[chunk_slice] = final_instances

                # Store boundary data for future chunks
                boundary_data[chunk_id] = {
                    'global_map': global_label_map,
                }

                # Create boundary-specific global mapping for labels that actually appear in boundaries
                boundary_global_map = {}

                # Store boundary regions if they exist and create boundary mappings
                if 'z_back' in overlap_regions:
                    z_back_region = final_instances[overlap_regions['z_back']]
                    boundary_data[chunk_id]['z_back'] = z_back_region
                    # Map boundary labels to their global equivalents
                    for label in np.unique(z_back_region):
                        if label > 0:
                            boundary_global_map[label] = label

                if 'y_back' in overlap_regions:
                    y_back_region = final_instances[overlap_regions['y_back']]
                    boundary_data[chunk_id]['y_back'] = y_back_region
                    for label in np.unique(y_back_region):
                        if label > 0:
                            boundary_global_map[label] = label

                if 'x_back' in overlap_regions:
                    x_back_region = final_instances[overlap_regions['x_back']]
                    boundary_data[chunk_id]['x_back'] = x_back_region
                    for label in np.unique(x_back_region):
                        if label > 0:
                            boundary_global_map[label] = label

                boundary_data[chunk_id]['boundary_global_map'] = boundary_global_map

                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"Processed {chunk_count}/{total_chunks} chunks")

    # Save TIFF if needed
    if output_format == "tiff":
        tff.imwrite(output_filepath, output_data.astype(np.uint32))

    print(f"Conversion complete! Output saved to: {output_filepath}")
    print(f"Total unique instances: {label_manager.next_global_id - 1}")


# --- Configuration for Conversion ---
class ConversionArgs:
    def __init__(self):
        # Path to your saved semantic prediction file (from inference script)
        self.input_prediction_path = "/media/samia/DATA/PhD/codebases/MitoEM/inference_results/merged_predictions_hemi_test.tif"  # <--- UPDATE THIS

        self.output_instance_dir = "instance_results"
        self.output_format = "tiff"  # "zarr" or "tiff"

        # Parameters for chunking
        self.chunk_size = (128, 128, 128)  # Size of chunks to process
        self.overlap = (16, 16, 16)  # Overlap between chunks (must be even for simple overlap)

        # Parameters for instance segmentation method (binary_connected)
        self.thres_foreground = 0.5  # Threshold for binarizing semantic prediction (0-1)
        self.thres_small_instances = 128  # Size threshold for removing small objects (pixels)
        self.scale_factors = (1.0, 1.0, 1.0)  # Keep at 1.0 for no resizing
        self.remove_small_mode = 'background'


if __name__ == "__main__":
    # Instantiate conversion arguments
    conv_args = ConversionArgs()

    # Run the conversion
    convert_semantic_to_instances_chunked(
        input_file_path=conv_args.input_prediction_path,
        output_dir=conv_args.output_instance_dir,
        output_format=conv_args.output_format,
        chunk_size=conv_args.chunk_size,
        overlap=conv_args.overlap,
        method="binary_connected",  # Only this method is supported without contours
        thres_foreground=conv_args.thres_foreground,
        thres_small_instances=conv_args.thres_small_instances,
        scale_factors=conv_args.scale_factors,
        remove_small_mode=conv_args.remove_small_mode
    )
