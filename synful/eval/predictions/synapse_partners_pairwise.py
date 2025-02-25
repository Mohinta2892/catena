from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import os


def distance(a, b):
    return np.linalg.norm(np.array(list(a)) - np.array(list(b)))


def cost(pre_post_location1, pre_post_location2, matching_threshold):
    max_cost = 2 * matching_threshold

    # Calculate distances exactly as in original
    pre_dist = distance(pre_post_location1[0], pre_post_location2[0])
    post_dist = distance(pre_post_location1[1], pre_post_location2[1])

    if pre_dist > matching_threshold or post_dist > matching_threshold:
        return max_cost

    return 0.5 * (pre_dist + post_dist)


def synaptic_partners_fscore(pred_pre_post_pairs, gt_pre_post_pairs, matching_threshold=400, all_stats=False):
    """Compute the f-score of the found synaptic partners using pairwise distances.
    
    Parameters
    ----------
    pred_pre_post_pairs: list of tuples
        Each tuple contains ((pre_x,pre_y,pre_z), (post_x,post_y,post_z))
    gt_pre_post_pairs: list of tuples
        Each tuple contains ((pre_x,pre_y,pre_z), (post_x,post_y,post_z))
    """
    # Initialize cost matrix
    n_pred = len(pred_pre_post_pairs)
    n_gt = len(gt_pre_post_pairs)
    size = max(n_pred, n_gt)
    costs = np.full((size, size), 2 * matching_threshold, dtype=float)

    # Calculate costs using pairwise distances
    for i in range(n_pred):
        for j in range(n_gt):
            costs[i, j] = cost(pred_pre_post_pairs[i], gt_pre_post_pairs[j], matching_threshold)

    # Find optimal matches using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(costs)

    # Filter matches within threshold
    filtered_matches = [(i, j, costs[i, j])
                        for i, j in zip(row_ind, col_ind)
                        if costs[i, j] <= matching_threshold]

    # Calculate metrics
    tp = len(filtered_matches)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = float(tp) / (tp + fp) if tp + fp > 0 else 0
    recall = float(tp) / (tp + fn) if tp + fn > 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    if all_stats:
        return fscore, precision, recall, fp, fn, filtered_matches
    return fscore


def main():
    # Read coordinates from CSV files
    data_path = "/Volumes/SamiaSan/PhD_Data/Synapses/synapse_predictions/octo_data_cube2"
    model_name = "model4_all3cubes/octo_cube2_pre_model4"
    gt_pre_df = pd.read_csv(os.path.join(data_path, 'gt',
                                         "octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_p/octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_pre_locations.csv"))
    gt_post_df = pd.read_csv(os.path.join(data_path, 'gt',
                                          "octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_p/octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_post_locations.csv"))
    pred_pre_df = pd.read_csv(os.path.join(data_path, model_name, "pred_pre_locations.csv"))
    pred_post_df = pd.read_csv(os.path.join(data_path, model_name, "pred_post_locations.csv"))

    # Create pre-post pairs in the correct format
    pred_pairs = [
        (
            (row_pre[1]['Pre_X'], row_pre[1]['Pre_Y'], row_pre[1]['Pre_Z']),
            (row_post[1]['Post_X'], row_post[1]['Post_Y'], row_post[1]['Post_Z'])
        )
        for row_pre, row_post in zip(pred_pre_df.iterrows(), pred_post_df.iterrows())
    ]

    gt_pairs = [
        (
            (row_pre[1]['Pre_X'], row_pre[1]['Pre_Y'], row_pre[1]['Pre_Z']),
            (row_post[1]['Post_X'], row_post[1]['Post_Y'], row_post[1]['Post_Z'])
        )
        for row_pre, row_post in zip(gt_pre_df.iterrows(), gt_post_df.iterrows())
    ]
    print(f"Number of gt_pairs: {len(gt_pairs)}")

    # Calculate F1 score
    fscore, precision, recall, fp, fn, matches = synaptic_partners_fscore(
        pred_pairs, gt_pairs, matching_threshold=550, all_stats=True
    )

    # Save matched pairs with their locations
    matched_pairs = []
    for pred_idx, gt_idx, dist in matches:
        matched_pairs.append({
            'pred_pre_x': pred_pairs[pred_idx][0][0],
            'pred_pre_y': pred_pairs[pred_idx][0][1],
            'pred_pre_z': pred_pairs[pred_idx][0][2],
            'pred_post_x': pred_pairs[pred_idx][1][0],
            'pred_post_y': pred_pairs[pred_idx][1][1],
            'pred_post_z': pred_pairs[pred_idx][1][2],
            'gt_pre_x': gt_pairs[gt_idx][0][0],
            'gt_pre_y': gt_pairs[gt_idx][0][1],
            'gt_pre_z': gt_pairs[gt_idx][0][2],
            'gt_post_x': gt_pairs[gt_idx][1][0],
            'gt_post_y': gt_pairs[gt_idx][1][1],
            'gt_post_z': gt_pairs[gt_idx][1][2],
            # 'cost((pre_dist+post_dist)/2)': dist  # this was distance before and is saved as distances in the csvs
            'distance': dist
        })

    out_dir = os.path.join(data_path, "results", model_name)
    os.makedirs(out_dir, exist_ok=True)  #  add a results dir if it is missing
    pd.DataFrame(matched_pairs).to_csv(f'{out_dir}/matched_pairs.csv', index=False)

    # Save unmatched predicted pairs (false positives)
    matched_pred_indices = set(m[0] for m in matches)
    unmatched_pred_pairs = []
    for idx, pair in enumerate(pred_pairs):
        if idx not in matched_pred_indices:
            unmatched_pred_pairs.append({
                'pred_pre_x': pair[0][0],
                'pred_pre_y': pair[0][1],
                'pred_pre_z': pair[0][2],
                'pred_post_x': pair[1][0],
                'pred_post_y': pair[1][1],
                'pred_post_z': pair[1][2]
            })
    pd.DataFrame(unmatched_pred_pairs).to_csv(f'{out_dir}/false_positives.csv', index=False)

    # Save unmatched ground truth pairs (false negatives)
    matched_gt_indices = set(m[1] for m in matches)
    unmatched_gt_pairs = []
    for idx, pair in enumerate(gt_pairs):
        if idx not in matched_gt_indices:
            unmatched_gt_pairs.append({
                'gt_pre_x': pair[0][0],
                'gt_pre_y': pair[0][1],
                'gt_pre_z': pair[0][2],
                'gt_post_x': pair[1][0],
                'gt_post_y': pair[1][1],
                'gt_post_z': pair[1][2]
            })
    pd.DataFrame(unmatched_gt_pairs).to_csv(f'{out_dir}/false_negatives.csv', index=False)

    print(f"F1 Score: {fscore:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {len(matches)}")

    with open(f"{os.path.join(out_dir, 'metrics.txt')}", "w") as f:
        f.write(f"Num of gt_pairs: {len(gt_pairs)}\n")
        f.write(f"F1 Score: {fscore:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {len(matches)}\n")

    # Save the matches
    matches_df = pd.DataFrame(matches, columns=['pred_idx', 'gt_idx', 'distance'])
    matches_df.to_csv(f'{out_dir}/synaptic_matches_pairwise.csv', index=False)


if __name__ == "__main__":
    main()
