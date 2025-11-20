from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


def synaptic_partners_fscore(pred_pre, pred_post, gt_pre, gt_post, matching_threshold=400, all_stats=False):
    """Compute the f-score of the found synaptic partners using KDTree.
    
    Parameters
    ----------
    pred_pre: array-like, shape (N, 3)
        Predicted pre-synaptic locations (x,y,z)
    pred_post: array-like, shape (N, 3)
        Predicted post-synaptic locations (x,y,z)
    gt_pre: array-like, shape (M, 3)
        Ground truth pre-synaptic locations (x,y,z)
    gt_post: array-like, shape (M, 3)
        Ground truth post-synaptic locations (x,y,z)
    matching_threshold: float
        Distance threshold for considering a match
    all_stats: boolean
        Whether to return detailed statistics
    """
    # Convert inputs to numpy arrays
    pred_pre = np.array(pred_pre)
    pred_post = np.array(pred_post)
    gt_pre = np.array(gt_pre)
    gt_post = np.array(gt_post)

    # Validate input shapes
    try:

        if len(pred_pre) != len(pred_post):
            raise ValueError(f"Number of predicted pre ({len(pred_pre)}) and post ({len(pred_post)}) synaptic points must match")
        if len(gt_pre) != len(gt_post):
            raise ValueError(f"Number of ground truth pre ({len(gt_pre)}) and post ({len(gt_post)}) synaptic points must match")    # Find the common indices between pre and post for both pred and gt

    except Exception as e:
        pred_min_len = min(len(pred_pre), len(pred_post))
        gt_min_len = min(len(gt_pre), len(gt_post))
        # Truncate to matching pairs
        pred_pre = pred_pre[:pred_min_len]
        pred_post = pred_post[:pred_min_len]
        gt_pre = gt_pre[:gt_min_len]
        gt_post = gt_post[:gt_min_len]


    # Combine pre and post coordinates into single 6D points
    pred_pairs = np.hstack([pred_pre, pred_post])  # Shape: (N, 6)
    gt_pairs = np.hstack([gt_pre, gt_post])        # Shape: (M, 6)
    
    # Build KD-tree for ground truth pairs
    gt_tree = cKDTree(gt_pairs)
    
    # Initialize cost matrix
    n_pred = len(pred_pairs)
    n_gt = len(gt_pairs)
    size = max(n_pred, n_gt)
    costs = np.full((size, size), 2 * matching_threshold, dtype=float)
    
    # Calculate costs using KDTree queries
    for i in range(n_pred):
        # Find distances to nearest ground truth pairs
        dists, indices = gt_tree.query(pred_pairs[i], k=n_gt)
        
        # Split the 6D distances into pre and post components
        pre_dists = np.linalg.norm(pred_pre[i] - gt_pre[indices], axis=1)
        post_dists = np.linalg.norm(pred_post[i] - gt_post[indices], axis=1)
        
        # Only consider it a match if both pre and post are within threshold
        valid_matches = (pre_dists <= matching_threshold) & (post_dists <= matching_threshold)
        avg_dists = 0.5 * (pre_dists + post_dists)
        costs[i, indices] = np.where(valid_matches, avg_dists, 2 * matching_threshold)

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
    gt_pre_df = pd.read_csv('data/gt_pre_locations.csv')
    gt_post_df = pd.read_csv('data/gt_post_locations.csv')
    pred_pre_df = pd.read_csv('data/pred_pre_locations.csv')
    pred_post_df = pd.read_csv('data/pred_post_locations.csv')

    print(f"Loaded data shapes:")
    print(f"GT Pre: {gt_pre_df.shape}, GT Post: {gt_post_df.shape}")
    print(f"Pred Pre: {pred_pre_df.shape}, Pred Post: {pred_post_df.shape}")

    # Convert to numpy arrays with correct format (x, y, z)
    gt_pre = gt_pre_df[['Pre_X', 'Pre_Y', 'Pre_Z']].values
    gt_post = gt_post_df[['Post_X', 'Post_Y', 'Post_Z']].values
    pred_pre = pred_pre_df[['Pre_X', 'Pre_Y', 'Pre_Z']].values
    pred_post = pred_post_df[['Post_X', 'Post_Y', 'Post_Z']].values

    # Calculate F1 score with detailed stats
    fscore, precision, recall, fp, fn, matches = synaptic_partners_fscore(
        pred_pre, pred_post, gt_pre, gt_post,
        matching_threshold=400, all_stats=True
    )

    print(f"F1 Score: {fscore:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {len(matches)}")

    # Optionally save the matches
    matches_df = pd.DataFrame(matches, columns=['pred_idx', 'gt_idx', 'distance'])
    matches_df.to_csv('results/synaptic_matches.csv', index=False)


if __name__ == "__main__":
    main()
