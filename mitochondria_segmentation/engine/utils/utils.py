import numpy as np
# Useful utils for post-processing
from scipy import ndimage
from skimage.morphology import remove_small_objects, dilation


# Count model parameters
def count_parameters(model):
    """
    Calculates the total number of trainable parameters in a PyTorch model.
    """
    # Sum the number of elements in each parameter tensor that requires gradients
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameters(model, model_name="Model"):
    """
    Prints the total number of parameters in a model in a human-readable format.
    """
    total_params = count_parameters(model)
    total_params = total_params / (1024 * 1024)
    print(f"{model_name} has ~{total_params:,}M trainable parameters.")

    # Optionally, print parameters layer by layer for more detail
    # print("\nParameters by layer:")
    # for name, parameter in model.named_parameters():
    #     if parameter.requires_grad:
    #         print(f"  {name}: {parameter.numel():,} parameters")


def binarize_and_median(pred, size=(7, 7, 7), thres=0.8):
    """First binarize the prediction with a given threshold, and
    then conduct median filtering to reduce noise.
    """
    pred = (pred > thres).astype(np.uint8)
    pred = ndimage.median_filter(pred, size=size)
    return pred


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
