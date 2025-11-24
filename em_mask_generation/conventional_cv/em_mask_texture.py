from tifffile import imread, imwrite
import numpy as np
from scipy import ndimage
from skimage import filters, feature, morphology
from typing import Tuple, Optional

data_path = "/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/Andre/code/sam-for-mask/data/s5_Parker.tif"
volume = imread(data_path)


def em_texture_masking(
        volume: np.ndarray,
        block_size: int = 31,
        texture_sigma: float = 2.0,
        edge_sigma: float = 1.0,
        chunk_size: Optional[Tuple[int, int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EM-specific masking using local texture and structure analysis.

    Parameters:
    -----------
    volume : ndarray
        3D input EM volume
    block_size : int
        Size of block for local statistics (odd number)
    texture_sigma : float
        Sigma for texture filtering
    edge_sigma : float
        Sigma for edge detection
    chunk_size : tuple, optional
        Size of chunks for processing (z, y, x)

    Returns:
    --------
    mask : ndarray
        Binary mask of the same shape as input
    confidence : ndarray
        Confidence map (0-1) indicating reliability of masking
    """

    def process_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize chunk
        chunk_norm = (chunk - np.percentile(chunk, 1)) / (
                np.percentile(chunk, 99) - np.percentile(chunk, 1)
        )
        chunk_norm = np.clip(chunk_norm, 0, 1)

        print(f"footprint: {morphology.disk(block_size // 2)}")
        # 1. Local texture analysis
        local_var = ndimage.variance(chunk_norm,
                                     index=morphology.disk(block_size // 2))  # footprint=morphology.disk(block_size//2)
        texture_mask = local_var > filters.threshold_otsu(local_var)

        # 2. Structure tensor analysis for oriented features
        Gx = ndimage.gaussian_filter1d(chunk_norm, texture_sigma, axis=1, order=1)
        Gy = ndimage.gaussian_filter1d(chunk_norm, texture_sigma, axis=0, order=1)

        Gxx = Gx * Gx
        Gyy = Gy * Gy
        Gxy = Gx * Gy

        # Compute coherence
        coherence = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2) / (Gxx + Gyy + 1e-6)
        coherence = ndimage.gaussian_filter(coherence, edge_sigma)

        # 3. Edge detection using Scharr operator
        edges = np.zeros_like(chunk_norm)
        for i in range(chunk_norm.shape[0]):
            edges[i] = filters.scharr(chunk_norm[i])
        edges = ndimage.gaussian_filter(edges, edge_sigma)

        # 4. Combine evidence
        combined_evidence = (
                0.4 * texture_mask +
                0.3 * (coherence > filters.threshold_otsu(coherence)) +
                0.3 * (edges > filters.threshold_otsu(edges))
        )

        # 5. Final mask with adaptive thresholding
        final_mask = combined_evidence > 0.5

        # 6. Clean up
        final_mask = morphology.remove_small_objects(final_mask, min_size=100)
        final_mask = morphology.remove_small_holes(final_mask, area_threshold=100)

        # Compute confidence based on evidence strength
        confidence = ndimage.gaussian_filter(combined_evidence, sigma=1.0)

        return final_mask, confidence

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


if __name__ == "__main__":
    # Basic usage
    mask, confidence = em_texture_masking(
        volume,
        block_size=31,  # Adjust based on feature size
        texture_sigma=2.0,
        edge_sigma=1.0
    )

    # write the mask file
    imwrite("./TEXTURE_mask_s5_Parker_v2.tiff", mask)
