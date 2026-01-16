import numpy as np
import skimage.segmentation
from skimage import measure
import zarr
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 30})


class AveragePrecision:
    def __init__(self, iou_range=(0.5, 1.0), ignore_index=0, min_instance_size=None):
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size

    def __call__(self, input, target):
        assert isinstance(input, np.ndarray) and isinstance(target, np.ndarray)
        assert input.ndim == target.ndim == 3

        target, target_instances = self._filter_instances(target)

        # recall, precision = self._roc_curve(predicted=input, target=target, target_instances=target_instances)
        #
        # print(recall, precision)

        return self._calculate_average_precision(input, target, target_instances)

    def _calculate_average_precision(self, predicted, target, target_instances):
        recall, precision = self._roc_curve(predicted, target, target_instances)
        recall.insert(0, 0.0)  # insert 0.0 at beginning of list
        recall.append(1.0)  # insert 1.0 at end of list
        precision.insert(0, 0.0)  # insert 0.0 at beginning of list
        precision.append(0.0)  # insert 0.0 at end of list
        # make the precision(recall) piece-wise constant and monotonically decreasing
        # by iterating backwards starting from the last precision value (0.0)
        # see: https://www.jeremyjordan.me/evaluating-image-segmentation-models/ e.g.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        # compute the area under precision recall curve by simple integration of piece-wise constant function
        ap = 0.0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i - 1]) * precision[i]
        return ap

    def _roc_curve(self, predicted, target, target_instances):
        ROC = []
        predicted, predicted_instances = self._filter_instances(predicted)

        # compute precision/recall curve points for various IoU values from a given range
        for min_iou in np.arange(self.iou_range[0], self.iou_range[1], 0.1):
            # initialize false negatives set
            false_negatives = set(target_instances)
            # initialize false positives set
            false_positives = set(predicted_instances)
            # initialize true positives set
            true_positives = set()

            for pred_label in predicted_instances:
                target_label = self._find_overlapping_target(pred_label, predicted, target, min_iou)
                if target_label is not None:
                    # update TP, FP and FN
                    if target_label == self.ignore_index:
                        # ignore if 'ignore_index' is the biggest overlapping
                        false_positives.discard(pred_label)
                    else:
                        true_positives.add(pred_label)
                        false_positives.discard(pred_label)
                        false_negatives.discard(target_label)

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            ROC.append((recall, precision))

        # sort points by recall
        ROC = np.array(sorted(ROC, key=lambda t: t[0]))
        # return recall and precision values
        return list(ROC[:, 0]), list(ROC[:, 1])

    def _find_overlapping_target(self, predicted_label, predicted, target, min_iou):
        """
        Return ground truth label which overlaps by at least 'min_iou' with a given input label 'p_label'
        or None if such ground truth label does not exist.
        """
        mask_predicted = predicted == predicted_label
        overlapping_labels = target[mask_predicted]
        labels, counts = np.unique(overlapping_labels, return_counts=True)
        # retrieve the biggest overlapping label
        target_label_ind = np.argmax(counts)
        target_label = labels[target_label_ind]
        # return target label if IoU greater than 'min_iou'; since we're starting from 0.5 IoU there might be
        # only one target label that fulfill this criterion
        mask_target = target == target_label
        # return target_label if IoU > min_iou
        if self._iou(mask_predicted, mask_target) > min_iou:
            return target_label
        return None

    @staticmethod
    def _iou(prediction, target):
        """
        Computes intersection over union
        """
        intersection = np.logical_and(prediction, target)
        union = np.logical_or(prediction, target)
        return np.sum(intersection) / np.sum(union)

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 'ignore_index'
        :param input: input instance segmentation
        :return: tuple: (instance segmentation with small instances filtered, set of unique labels without the 'ignore_index')
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    mask = input == label
                    input[mask] = self.ignore_index

        labels = set(np.unique(input))
        labels.discard(self.ignore_index)
        return input, labels

    @staticmethod
    def _dt_to_cc(distance_transform, threshold):
        """
        Threshold a given distance_transform and returns connected components.
        :param distance_transform: 3D distance transform matrix
        :param threshold: threshold energy level
        :return: 3D segmentation volume
        """
        boundary = (distance_transform > threshold).astype(np.uint8)
        return measure.label(boundary, background=0, connectivity=1)


def ap(segmentation, ground_truth, ignore_index=0, min_instance_size=10000):
    """
    Returns Average Precision score between the given segmentation and the ground_truth segmentation.

    Args:
        segmentation (ndarray): input segmentation
        ground_truth (ndarray): ground truth segmentation
        ignore_index (int): label to be ignored during AP computation
        min_instance_size (int): minimum instance size used for AP computation; use in order to make the metric robust
            to small instances present either in the input or ground truth segmentation; if 'None' all instances are
            taken into account during AP computation
    Returns:
        Average Precision between segmentation and ground_truth
    """
    ap = AveragePrecision(ignore_index=ignore_index, min_instance_size=min_instance_size)
    return ap(segmentation, ground_truth)


def roc(predicted, target, ignore_index=0, min_instance_size=100):
    """
    Returns ROC recall precision based on the IOU range provided
    for the given segmentation and the ground_truth segmentation.

    Args:
        segmentation (ndarray): input segmentation
        ground_truth (ndarray): ground truth segmentation
        ignore_index (int): label to be ignored during AP computation
        min_instance_size (int): minimum instance size used for AP computation; use in order to make the metric robust
            to small instances present either in the input or ground truth segmentation; if 'None' all instances are
            taken into account during AP computation
    Returns:
        ROC recall and precision
    """
    ap = AveragePrecision(ignore_index=ignore_index, min_instance_size=min_instance_size)
    target, target_instances = ap._filter_instances(target)

    recall, precision = ap._roc_curve(predicted=predicted, target=target, target_instances=target_instances)

    # IOU thresholds from 0.5 to 0.9 with step size 0.1
    iou_thresholds = np.arange(0.5, 1.0, 0.1)

    return recall, precision

    # print(f"recall {recall}, precision {precision}")


def plot_roc(recall_dict, precision_dict, title="Hemi-Brain"):
    # IOU thresholds for plotting
    iou_thresholds = np.arange(0.5, 1.0, 0.1)

    # Plotting
    plt.figure(figsize=(15, 10))

    # Colors for different agglomeration thresholds
    colors = ['#377eb8', '#ff7f00', '#4daf4a']

    # Plot recall for different thresholds
    for i, (threshold, recall) in enumerate(recall_dict.items()):
        plt.plot(iou_thresholds, recall, marker='o', linestyle='-', color=colors[i], label=f'Recall {threshold}')

    # Plot precision for different thresholds
    for i, (threshold, precision) in enumerate(precision_dict.items()):
        plt.plot(iou_thresholds, precision, marker='x', linestyle='--', color=colors[i], label=f'Precision {threshold}')

    # Add labels and title
    plt.xlabel('IOU Threshold', )  # fontsize=16)
    plt.ylabel('Value', )  # fontsize=16)
    # plt.title('Recall and Precision across IOU Thresholds for Different Agglomeration Thresholds')
    plt.title(title, )  # fontsize=18)
    plt.ylim((0, 1.1))
    # plt.legend()

    # Adding grid for better readability
    plt.grid(True)
    sns.despine()

    # Increase the font size of the legend
    plt.legend()  # fontsize=14

    # Increase tick parameters for better readability
    plt.xticks()  # fontsize=14
    plt.yticks()  # fontsize=14

    plt.savefig(f"./roc_{title}.png", dpi=300)

    # Show plot
    plt.show()


def main():
    # for now let's make a main here to test
    # hemibrain
    f = zarr.open(
        "/media/samia/DATA/ark/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_270000/roi_1_full_remapped.zarr")
    gt_bin = f["volumes/binary_affs_roi1"][:20, :200, :200]  # remember has to be a small volume to fit in memory
    pred_bin = f["volumes/binary_pred_affs_roi1"][:20, :200, :200]
    segmentation1 = f["volumes/segmentation_05"][:20, :200, :200]
    segmentation2 = f["volumes/segmentation_055"][:20, :200, :200]
    segmentation3 = f["volumes/segmentation_06"][:20, :200, :200]
    ground_truth, _, _ = skimage.segmentation.relabel_sequential(f["volumes/labels/neuron_ids_roi1"][:20, :200, :200])
    avg_prec_obj = AveragePrecision(iou_range=[0.5, 1], ignore_index=0)
    iou = avg_prec_obj._iou(pred_bin, gt_bin)
    recall_1, prec_1 = roc(segmentation1, ground_truth)
    recall_2, prec_2 = roc(segmentation2, ground_truth)
    recall_3, prec_3 = roc(segmentation3, ground_truth)
    recall_dict = {"50": recall_1, "55": recall_2, "60": recall_3}

    precision_dict = {"50": prec_1, "55": prec_2, "60": prec_3}

    print(f"IOU {iou}")
    print(f"Average Prec {ap(segmentation1, ground_truth)}")
    print(f"ROC {recall_dict, precision_dict}")
    plot_roc(recall_dict, precision_dict)

    # octo
    focto_gt = zarr.open(
        "/home/samia/Downloads/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt1.zarr")
    focto = zarr.open(
        "/home/samia/Downloads/otto_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_1.zarr")
    segmentation_octo1 = focto["volumes/segmentation_05"][...]
    segmentation_octo2 = focto["volumes/segmentation_055"][...]
    segmentation_octo3 = focto["volumes/segmentation_06"][...]
    ground_truth_octo = focto_gt["volumes/labels/neuron_ids"][...]
    recall_1, prec_1 = roc(segmentation_octo1, ground_truth_octo)
    recall_2, prec_2 = roc(segmentation_octo2, ground_truth_octo)
    recall_3, prec_3 = roc(segmentation_octo3, ground_truth_octo)
    recall_dict = {"50": recall_1, "55": recall_2, "60": recall_3}

    precision_dict = {"50": prec_1, "55": prec_2, "60": prec_3}
    plot_roc(recall_dict, precision_dict, title="Octo with Synthetic")
    # print(f"ROC {roc(segmentation_octo, ground_truth_octo)}")

    focto_gt = zarr.open(
        "/home/samia/Downloads/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt1.zarr")
    focto_previous = zarr.open(
        "/media/samia/DATA/ark/dan-samia/lsd/funke/otto/tiff/octo_z7392-7904_y6586-7098_x5388-5900_z120-140_y100-300_x100-300_wgt.zarr")
    segmentation_octo1 = focto_previous["volumes/segmentation_0.5"][...]
    segmentation_octo2 = focto_previous["volumes/segmentation_0.55"][...]
    segmentation_octo3 = focto_previous["volumes/segmentation_0.6"][...]
    ground_truth_octo = focto_gt["volumes/labels/neuron_ids"][...]
    recall_1, prec_1 = roc(segmentation_octo1, ground_truth_octo)
    recall_2, prec_2 = roc(segmentation_octo2, ground_truth_octo)
    recall_3, prec_3 = roc(segmentation_octo3, ground_truth_octo)
    recall_dict = {"50": recall_1, "55": recall_2, "60": recall_3}

    precision_dict = {"50": prec_1, "55": prec_2, "60": prec_3}
    plot_roc(recall_dict, precision_dict, title="Octo without Synthetic")


if __name__ == "__main__":
    main()
