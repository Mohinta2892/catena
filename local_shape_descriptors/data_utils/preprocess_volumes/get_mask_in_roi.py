import numpy as np
from funlib.persistence import Array as fpArray


def upsample(a, factor):
    """For mask upsampling."""
    for d, f in enumerate(factor):
        a = np.repeat(a, f, axis=d)

    return a


def get_mask_data_in_roi(mask, roi, target_voxel_size):
    """Upsample a smaller mask to the larger roi and align it """
    assert mask.voxel_size.is_multiple_of(target_voxel_size), (
            "Can not upsample from %s to %s" % (mask.voxel_size, target_voxel_size))

    aligned_roi = roi.snap_to_grid(mask.voxel_size, mode='grow')
    aligned_data = mask.to_ndarray(aligned_roi, fill_value=0)

    if mask.voxel_size == target_voxel_size:
        return aligned_data

    factor = mask.voxel_size / target_voxel_size

    upsampled_aligned_data = upsample(aligned_data, factor)

    # this is funlib.persistence array
    upsampled_aligned_mask = fpArray(
        upsampled_aligned_data,
        roi=aligned_roi,
        voxel_size=target_voxel_size)

    return upsampled_aligned_mask.to_ndarray(roi)
