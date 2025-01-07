from scipy.optimize import linear_sum_assignment
import numpy as np
import logging

logger = logging.getLogger(__name__)


def synaptic_partners_fscore(rec_annotations, gt_annotations,
                             matching_threshold=400,
                             all_stats=False, use_only_pre=False,
                             use_only_post=False,
                             id_type='skel'):
    """Compute the f-score of the found synaptic partners. Original function
    from: https://github.com/cremi/cremi_python. Modified, such that it
    works with synful.Synapse. Represents fast version, in which connectivity pairs
    are already preselected, before solving hungarian matching.

    Parameters
    ----------

    rec_annotations: List of predicted synapses (synful.Synapse)

    gt_annotations: List of ground truth synapses (synful.Synapse)

    matching_threshold: float, world units
        Euclidean distance threshold to consider two synapses a potential
        match. Synapses that are `matching_threshold` or more untis apart
        from each other are not considered as potential matches.

    use_only_pre: whether to only consider the distance of presites for
        applying the matching_threshold.

    use_only_post: whether to only consider the distance of postsites for
        applying the matching_threshold.

    all_stats: boolean, optional
        Whether to also return precision, recall, FP, FN, and matches as a 6-tuple with f-score

    id_type: str, optional
        Whether to use segmentation id (seg) or skeleton id (skel) for
        checking connectivity.

    Returns
    -------

    fscore: float
        The f-score of the found synaptic partners.
    precision: float, optional
    recall: float, optional
    fp: int, optional
    fn: int, optional
    all_syns: (tp_syns, fp_syns, fn_syns_gt, tp_syns_gt)
    """

    # Help hungarian matching by prematching synapses with same connectivity
    gt_pair_dic = {}
    for syn in gt_annotations:
        if id_type == 'seg':
            pair = (syn.id_segm_pre, syn.id_segm_post)
        elif id_type == 'skel':
            pair = (syn.id_skel_pre, syn.id_skel_post)
        else:
            raise Exception('id_type {} not known'.format(id_type))
        gt_pair_dic.setdefault(pair, [])
        gt_pair_dic[pair].append(syn)

    rec_pair_dic = {}
    for syn in rec_annotations:
        if id_type == 'seg':
            pair = (syn.id_segm_pre, syn.id_segm_post)
        elif id_type == 'skel':
            pair = (syn.id_skel_pre, syn.id_skel_post)
        else:
            raise Exception('id_type {} not known'.format(id_type))
        rec_pair_dic.setdefault(pair, [])
        rec_pair_dic[pair].append(syn)

    fpcountall = 0
    fncountall = 0
    all_matches = []
    fscores = []
    tp_syns_all = []
    fp_syns_all = []
    fn_syns_gt_all = []
    tp_syns_gt_all = []
    for pair, gt_syns in gt_pair_dic.items():
        pred_syns = rec_pair_dic.get(pair, [])
        if len(pred_syns) != 0:
            fscore, precision, recall, fpcount, fncount, filtered_matches = __synaptic_partners_fscore(
                pred_syns, gt_syns, matching_threshold,
                all_stats=True, use_only_pre=use_only_pre,
                use_only_post=use_only_post, id_type=id_type)
            tp_syns, fp_syns, fn_syns_gt, tp_syns_gt = from_synapsematches_to_syns(
                filtered_matches, pred_syns, gt_syns)
            tp_syns_all.extend(tp_syns)
            fp_syns_all.extend(fp_syns)
            fn_syns_gt_all.extend(fn_syns_gt)
            tp_syns_gt_all.extend(tp_syns_gt)
            fscores.append(fscore)
        else:
            # If no predicted synapse with gt_pair connectivity, they are all FNs.
            fncount = len(
                gt_syns)
            fpcount = 0
            filtered_matches = []
            fn_syns_gt_all.extend(gt_syns)

        fpcountall += fpcount
        fncountall += fncount
        all_matches.extend(filtered_matches)

    # All pairs that are not in the ground truth represent False Positives
    unmatched_pairs = set(rec_pair_dic.keys()) - set(gt_pair_dic.keys())
    for pair in list(unmatched_pairs):
        fpcountall += len(rec_pair_dic[pair])
        fp_syns_all.extend(rec_pair_dic[pair])
    tp = len(all_matches)
    fp = fpcountall
    fn = fncountall

    assert fncountall == len(fn_syns_gt_all)
    assert fpcountall == len(fp_syns_all)
    assert len(tp_syns_gt_all) == len(tp_syns_all)
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 1.
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 1.
    if (precision + recall) > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    if all_stats:

        all_syns = [tp_syns_all, fp_syns_all, fn_syns_gt_all, tp_syns_gt_all]
        return (fscore, precision, recall, fp, fn, all_syns)
    else:
        return fscore


def __synaptic_partners_fscore(rec_annotations, gt_annotations, matching_threshold=400,
                             all_stats=False, use_only_pre=False, use_only_post=False,
                             id_type='skel'):
    """Compute the f-score of the found synaptic partners. Original function
    from: https://github.com/cremi/cremi_python. Modified, such that it
    works with synful.Synapse.

    Parameters
    ----------

    rec_annotations: List of predicted synapses (synful.Synapse)

    gt_annotations: List of ground truth synapses (synful.Synapse)

    matching_threshold: float, world units
        Euclidean distance threshold to consider two synapses a potential
        match. Synapses that are `matching_threshold` or more untis apart
        from each other are not considered as potential matches.

    use_only_pre: whether to only consider the distance of presites for
        applying the matching_threshold.

    use_only_post: whether to only consider the distance of postsites for
        applying the matching_threshold.

    all_stats: boolean, optional
        Whether to also return precision, recall, FP, FN, and matches as a 6-tuple with f-score

    id_type: str, optional
        Whether to use segmentation id (seg) or skeleton id (skel) for
        checking connectivity.

    Returns
    -------

    fscore: float
        The f-score of the found synaptic partners.
    precision: float, optional
    recall: float, optional
    fp: int, optional
    fn: int, optional
    filtered_matches: list of tuples, optional
        The indices of the matches with matching costs.
    """

    # get cost matrix
    costs = cost_matrix(rec_annotations, gt_annotations, matching_threshold,
                        id_type, use_only_pre, use_only_post)

    # match using Hungarian method
    logger.debug("Finding cost-minimal matches...")
    matches = linear_sum_assignment(costs - np.amax(costs) - 1)
    matches = zip(matches[0], matches[1])  # scipy returns matches as numpy arrays

    filtered_matches = [(i, j, costs[i][j]) for (i, j) in matches if costs[i][j] <= matching_threshold]
    logger.debug(str(len(filtered_matches)) + " matches found")

    # unmatched in rec = FP
    fp = len(rec_annotations) - len(filtered_matches)

    # unmatched in gt = FN
    fn = len(gt_annotations) - len(filtered_matches)

    # all ground truth elements - FN = TP
    tp = len(gt_annotations) - fn

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    if (precision + recall) > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    if all_stats:
        return (fscore, precision, recall, fp, fn, filtered_matches)
    else:
        return fscore

def from_synapsematches_to_syns(matches, pred_synapses, gt_synapses):

    # True Positives PRED
    tp_ids_test = list(zip(*matches))[0] if len(matches) > 0 else []
    tp_ids_truth = list(zip(*matches))[1] if len(matches) > 0 else []
    tp_syns = [pred_synapses[index1] for index1 in tp_ids_test]

    # False Positives.
    test_ids = set(range(len(pred_synapses)))
    false_positive_ids = test_ids - set(tp_ids_test)
    fp_syns = [pred_synapses[ii] for ii in list(false_positive_ids)]

    # False Negative.
    truth_ids = set(range(len(gt_synapses)))
    false_negative_ids = truth_ids - set(tp_ids_truth)
    fn_syns_gt = [gt_synapses[ii] for ii in list(false_negative_ids)]

    # True Positives GT
    tp_syns_gt = [gt_synapses[index1] for index1 in tp_ids_truth]

    return tp_syns, fp_syns, fn_syns_gt, tp_syns_gt

def cost_matrix(rec, gt, matching_threshold, id_type, use_only_pre=False, use_only_post=False):
    logger.debug("Computing matching costs...")

    rec_locations = [(syn.location_pre, syn.location_post) for syn in rec]
    gt_locations = [(syn.location_pre, syn.location_post) for syn in gt]

    size = max(len(rec_locations), len(gt_locations))
    costs = np.zeros((size, size), dtype=np.float)
    costs[:] = 2 * matching_threshold
    num_potential_matches = 0
    for i in range(len(rec_locations)):
        for j in range(len(gt_locations)):
            c = cost(rec_locations[i], gt_locations[j], rec[i], gt[j], matching_threshold, use_only_pre, use_only_post, id_type)
            costs[i, j] = c
            if c <= matching_threshold:
                num_potential_matches += 1

    logger.debug(str(num_potential_matches) + " potential matches found")

    return costs


def cost(pre_post_location1, pre_post_location2, syn1, syn2,
         matching_threshold, use_only_pre, use_only_post, id_type):
    max_cost = 2 * matching_threshold

    # First check of the nodes are part of the same segment

    if id_type == 'skel':
        pre_label_same = syn1.id_skel_pre == syn2.id_skel_pre
        post_label_same = syn1.id_skel_post == syn2.id_skel_post
        if syn1.id_skel_pre is None or syn2.id_skel_pre is None:
            pre_label_same = False
        if syn1.id_skel_post is None or syn2.id_skel_post is None:
            post_label_same = False
    elif id_type == 'seg':
        pre_label_same = syn1.id_segm_pre == syn2.id_segm_pre
        post_label_same = syn1.id_segm_post == syn2.id_segm_post
        if syn1.id_segm_pre is None or syn2.id_segm_pre is None:
            pre_label_same = False
        if syn1.id_segm_post is None or syn2.id_segm_post is None:
            post_label_same = False
    else:
        raise ValueError('id_type {} unknown'.format(id_type))

    # pairs do not link the same segments
    if not pre_label_same or not post_label_same:
        return max_cost

    pre_dist = distance(pre_post_location1[0], pre_post_location2[0])
    post_dist = distance(pre_post_location1[1], pre_post_location2[1])
    if use_only_pre:
        if pre_dist > matching_threshold:
            return max_cost
        dist = pre_dist
    elif use_only_post:
        if post_dist > matching_threshold:
            return max_cost
        dist = post_dist
    else:
        if pre_dist > matching_threshold or post_dist > matching_threshold:
            return max_cost
        dist = 0.5 * (pre_dist + post_dist)

    return dist


def distance(a, b):
    return np.linalg.norm(np.array(list(a)) - np.array(list(b)))


def add(a, b):
    return tuple([a[d] + b[d] for d in range(len(b))])


def sub(a, b):
    return tuple([a[d] - b[d] for d in range(len(b))])
