import numpy as np
from scipy import ndimage
from skimage import filters, feature, morphology, measure
from skimage import morphology, segmentation, feature
import numpy.typing as npt
from typing import Tuple, Optional
import cv2
from tifffile import imread, imwrite

data_path = "/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/Andre/code/sam-for-mask/data/s5_Parker.tif"
volume = imread(data_path)


def create_filled_outline(
        mask: npt.NDArray,
        outline_width: int = 2,
        inside_value: int = 255,
        outside_value: int = 0
) -> npt.NDArray:
    """
    Create a filled mask based on the outline of an input mask.

    Parameters:
    -----------
    mask : NDArray
        Input binary mask
    outline_width : int
        Width of the outline to consider
    inside_value : int
        Value to fill inside the outline
    outside_value : int
        Value for regions outside the outline

    Returns:
    --------
    filled_mask : NDArray
        New mask with filled outline
    """
    # Initialize output with outside value
    filled_mask = np.full_like(mask, outside_value, dtype=np.uint8)

    # Process each slice for 3D volumes
    for z in range(mask.shape[0]):
        # Get current slice
        slice_mask = mask[z].astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            slice_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Fill contours
        if contours:
            cv2.drawContours(
                filled_mask[z],
                contours,
                -1,
                inside_value,
                -1  # Fill interior
            )

    return filled_mask


def filter_artifacts(
        mask: np.ndarray,
        min_size: int = 10000,
        max_eccentricity: float = 0.85,
        center_region_size: float = 0.15
) -> np.ndarray:
    """
    Filter out artifacts in 3D mask based on size, shape, and position

    Parameters:
    -----------
    mask : ndarray
        3D binary mask
    min_size : int
        Minimum region size in pixels per slice
    max_eccentricity : float
        Maximum eccentricity (0-1) for circular filtering
    center_region_size : float
        Size of center region to check (as fraction of image size)
    """
    clean_mask = np.zeros_like(mask, dtype=bool)

    # Process each slice
    for z in range(mask.shape[0]):
        # Get current slice
        slice_mask = mask[z]

        # Get slice center and dimensions
        center_y, center_x = np.array(slice_mask.shape) // 2
        h, w = slice_mask.shape
        center_region_h = int(h * center_region_size)
        center_region_w = int(w * center_region_size)

        # Label connected components in slice
        labels = measure.label(slice_mask)
        regions = measure.regionprops(labels)

        for region in regions:
            # Skip small regions
            if region.area < min_size:
                continue

            # Skip highly eccentric (non-circular) regions
            if region.eccentricity > max_eccentricity:
                continue

            # Check if region is entirely within center region
            cy, cx = region.centroid
            in_center = (
                    abs(cy - center_y) < center_region_h / 2 and
                    abs(cx - center_x) < center_region_w / 2 and
                    region.area < min_size * 2  # Smaller threshold for center objects
            )

            if not in_center:
                clean_mask[z][labels == region.label] = True

    # Optional: Add 3D connectivity check
    if mask.shape[0] > 1:  # Only if we have multiple slices
        # Label 3D connected components
        labels_3d = measure.label(clean_mask, connectivity=1)
        regions_3d = measure.regionprops(labels_3d)

        # Filter small 3D regions
        min_volume = min_size * 2  # Adjust this multiplier as needed
        for region in regions_3d:
            if region.area < min_volume:
                clean_mask[labels_3d == region.label] = False

    return clean_mask


def em_texture_masking(
        volume: np.ndarray,
        block_size: int = 31,
        texture_sigma: float = 2.0,
        edge_sigma: float = 1.0,
        min_region_size: int = 10000,
        chunk_size: Optional[Tuple[int, int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EM-specific masking with artifact removal

    Parameters:
    -----------
    volume : ndarray
        3D input EM volume
    block_size : int
        Size of block for local statistics
    texture_sigma : float
        Sigma for texture filtering
    edge_sigma : float
        Sigma for edge detection
    min_region_size : int
        Minimum size for valid regions
    chunk_size : tuple, optional
        Size of chunks for processing
    """

    def process_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize chunk using robust statistics
        p1, p99 = np.percentile(chunk, (1, 99))
        chunk_norm = np.clip((chunk - p1) / (p99 - p1), 0, 1)

        # 1. Enhanced texture analysis
        local_var = ndimage.variance(chunk_norm,
                                     index=morphology.disk(block_size // 2))
        texture_mask = local_var > filters.threshold_otsu(local_var)

        # 2. Structure tensor analysis
        Gx = ndimage.gaussian_filter1d(chunk_norm, texture_sigma, axis=1, order=1)
        Gy = ndimage.gaussian_filter1d(chunk_norm, texture_sigma, axis=0, order=1)

        Gxx = Gx * Gx
        Gyy = Gy * Gy
        Gxy = Gx * Gy

        coherence = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2) / (Gxx + Gyy + 1e-6)
        coherence = ndimage.gaussian_filter(coherence, edge_sigma)

        # 3. Edge detection
        edges = filters.scharr(chunk_norm)
        edges = ndimage.gaussian_filter(edges, edge_sigma)

        # 4. Combine evidence
        combined_evidence = (
                0.4 * texture_mask +
                0.3 * (coherence > filters.threshold_otsu(coherence)) +
                0.3 * (edges > filters.threshold_otsu(edges))
        )

        # 5. Initial mask
        initial_mask = combined_evidence > 0.5

        # # 6. Filter artifacts
        # clean_mask = filter_artifacts(
        #     initial_mask,
        #     min_size=min_region_size,
        #     max_eccentricity=0.85,
        #     center_region_size=0.15
        # )
        clean_mask = initial_mask

        # 7. Compute confidence
        confidence = ndimage.gaussian_filter(combined_evidence, sigma=1.0)
        confidence[~clean_mask] = 0  # Zero confidence in filtered regions

        return clean_mask, confidence

    # Handle chunking
    if chunk_size is None:
        chunk_size = (
            min(64, volume.shape[0]),
            min(512, volume.shape[1]),
            min(512, volume.shape[2])
        )

    # Initialize output arrays
    mask = np.zeros_like(volume, dtype=bool)
    confidence = np.zeros_like(volume, dtype=np.float32)

    # Process chunks
    for z in range(0, volume.shape[0], chunk_size[0]):
        z_end = min(z + chunk_size[0], volume.shape[0])
        for y in range(0, volume.shape[1], chunk_size[1]):
            y_end = min(y + chunk_size[1], volume.shape[1])
            for x in range(0, volume.shape[2], chunk_size[2]):
                x_end = min(x + chunk_size[2], volume.shape[2])

                chunk = volume[z:z_end, y:y_end, x:x_end]
                chunk_mask, chunk_conf = process_chunk(chunk)

                mask[z:z_end, y:y_end, x:x_end] = chunk_mask
                confidence[z:z_end, y:y_end, x:x_end] = chunk_conf

    return mask, confidence


if __name__ == '__main__':
    # Basic usage with artifact filtering
    mask, confidence = em_texture_masking(
        volume,
        block_size=64,
        texture_sigma=1.0,
        edge_sigma=0.5,
        min_region_size=250  # Adjust based on your image size
    )
    # Filter any artefacts
    # clean_mask = filter_artifacts(
    #             mask,
    #             min_size=250,
    #             max_eccentricity=0.85,
    #             center_region_size=0.15
    #         )

    # create filled outline
    filled_mask = create_filled_outline(mask)

    # write the mask file
    imwrite("./TEXTURE_mask_s5_Parker_v3.tiff", filled_mask)

    # Refill mask if multiple holes remain
    filled_mask = create_filled_outline(filled_mask)

    # write the new mask
    imwrite("./TEXTURE_mask_filled.tiff", filled_mask)