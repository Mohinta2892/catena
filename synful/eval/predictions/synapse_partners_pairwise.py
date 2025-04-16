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


def get_coordinate_columns(df, prefix=''):
    """Helper function to identify coordinate columns regardless of naming convention"""
    # First try to find ID column with various naming conventions
    id_col = None
    id_variants = [f'{prefix}_ID', f'{prefix}_id', f'{prefix.upper()}_ID', f'{prefix.lower()}_id']
    for variant in id_variants:
        if variant in df.columns:
            id_col = variant
            break

    # Then find coordinate columns
    if all(f'{prefix}_{ax}' in df.columns for ax in ['X', 'Y', 'Z']):
        coord_cols = [f'{prefix}_X', f'{prefix}_Y', f'{prefix}_Z']
    elif all(f'axis-{i}' in df.columns for i in [2, 1, 0]):  # axis-2=x, axis-1=y, axis-0=z
        coord_cols = ['axis-2', 'axis-1', 'axis-0']
    else:
        raise ValueError(f"Cannot find coordinate columns in DataFrame. Expected either {prefix}_X/Y/Z or axis-2/1/0")
    
    # Return both coordinate columns and ID column if found
    if id_col:
        return coord_cols, id_col
    return coord_cols, None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate matching statistics between predicted and ground truth synaptic partners.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
    python synapse_partners_pairwise.py \\
        --gt-pre gt_pre_locations.csv \\
        --gt-post gt_post_locations.csv \\
        --pred-pre pred_pre_locations.csv \\
        --pred-post pred_post_locations.csv \\
        --matching-threshold 550 \\
        --output-dir ./results
        
Input CSVs should contain columns: Pre_X, Pre_Y, Pre_Z for pre-synaptic and Post_X, Post_Y, Post_Z for post-synaptic locations.
        '''
    )

    parser.add_argument('--gt-pre', required=True,
                        help='CSV file containing ground truth pre-synaptic locations')
    parser.add_argument('--gt-post', required=True,
                        help='CSV file containing ground truth post-synaptic locations')
    parser.add_argument('--pred-pre', required=True,
                        help='CSV file containing predicted pre-synaptic locations')
    parser.add_argument('--pred-post', required=True,
                        help='CSV file containing predicted post-synaptic locations')
    parser.add_argument('--matching-threshold', type=float, default=550,
                        help='Distance threshold for matching synapses (default: 550nm). Adjust to pixel space when using pixel space coordinates.')
    parser.add_argument('--output-dir', default='./results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--pred-mapping-csv', default=None,
                      help='Optional CSV file containing pre_id,post_id mapping. Use -1 in post_id to indicate unpaired pre-synaptic points')
    parser.add_argument('--resolution-x', type=float, default=None,
                      help='Resolution in nm per pixel for X dimension (for conversion from pixel to nm space)')
    parser.add_argument('--resolution-y', type=float, default=None,
                      help='Resolution in nm per pixel for Y dimension (for conversion from pixel to nm space)')
    parser.add_argument('--resolution-z', type=float, default=None,
                      help='Resolution in nm per pixel for Z dimension (for conversion from pixel to nm space)')

    args = parser.parse_args()

    # Read coordinates from CSV files
    gt_pre_df = pd.read_csv(args.gt_pre)
    gt_post_df = pd.read_csv(args.gt_post)
    pred_pre_df = pd.read_csv(args.pred_pre)
    pred_post_df = pd.read_csv(args.pred_post)

    # Convert pixel coordinates to nm if resolution is provided
    if args.resolution_x is not None and args.resolution_y is not None and args.resolution_z is not None:
        print(f"Converting pixel coordinates to nm using resolutions: X={args.resolution_x}nm, Y={args.resolution_y}nm, Z={args.resolution_z}nm")
        
        # Function to convert dataframe columns
        def convert_to_nm(df, prefix):
            cols, _ = get_coordinate_columns(df, prefix)
            # Determine which axis is which (X, Y, Z)
            if prefix + '_X' in cols:
                x_col, y_col, z_col = prefix + '_X', prefix + '_Y', prefix + '_Z'
            else:  # axis-2, axis-1, axis-0
                x_col, y_col, z_col = 'axis-2', 'axis-1', 'axis-0'
                
            # Convert each coordinate
            df[x_col] = df[x_col] * args.resolution_x
            df[y_col] = df[y_col] * args.resolution_y
            df[z_col] = df[z_col] * args.resolution_z
            return df
        
        # Convert all dataframes
        pred_pre_df = convert_to_nm(pred_pre_df, 'Pre')
        pred_post_df = convert_to_nm(pred_post_df, 'Post')

    # Create output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Rest of the code remains the same, just replace the output paths with args.output_dir
    # For example:
    # pd.DataFrame(matched_pairs).to_csv(os.path.join(args.output_dir, 'matched_pairs.csv'), index=False)
    # data_path = "/Volumes/SamiaSan/PhD_Data/Synapses/synapse_predictions/octo_data_cube2"
    # model_name = "model4_all3cubes/octo_cube2_pre_model4"
    # gt_pre_df = pd.read_csv(os.path.join(data_path, 'gt',
    #                                      "octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_p/octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_pre_locations.csv"))
    # gt_post_df = pd.read_csv(os.path.join(data_path, 'gt',
    #                                       "octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_p/octo_cube2_12485_13164_y6231_6901_z3971_4640_gt_post_locations.csv"))
    # pred_pre_df = pd.read_csv(os.path.join(data_path, model_name, "pred_pre_locations.csv"))
    # pred_post_df = pd.read_csv(os.path.join(data_path, model_name, "pred_post_locations.csv"))

    # Create pre-post pairs in the correct format
    pred_pre_cols, pred_pre_id = get_coordinate_columns(pred_pre_df, 'Pre')
    pred_post_cols, pred_post_id = get_coordinate_columns(pred_post_df, 'Post')
    gt_pre_cols, gt_pre_id = get_coordinate_columns(gt_pre_df, 'Pre')
    gt_post_cols, gt_post_id = get_coordinate_columns(gt_post_df, 'Post')

    if args.pred_mapping_csv:
        mapping_df = pd.read_csv(args.pred_mapping_csv)

        # Determine column names in mapping file (case-insensitive)
        pre_id_col = None
        post_id_col = None
        for col in mapping_df.columns:
            if col.lower() in ['pre_id', 'preid', 'pre_id', 'Pre_ID']:
                pre_id_col = col
            elif col.lower() in ['post_id', 'postid', 'post_id', 'Post_ID']:
                post_id_col = col
        
        if not pre_id_col or not post_id_col:
            raise ValueError(f"Cannot find pre_id and post_id columns in mapping CSV. Found columns: {mapping_df.columns}")

        pred_pairs = []
        for _, row in mapping_df.iterrows():
            # Find the pre-synaptic point using ID if available, otherwise use index
            if pred_pre_id:
                try:
                    pre_row = pred_pre_df[pred_pre_df[pred_pre_id] == row[pre_id_col]].iloc[0]
                except Exception as e:
                    print(f"Warning: Could not find pre_id {row[pre_id_col]} in pred_pre_df")
                    continue
            else:
                try:
                    pre_row = pred_pre_df.iloc[row[pre_id_col]]
                except Exception as e:
                    print(f"Warning: Invalid pre_id index {row[pre_id_col]}")
                    continue
            
            pre_point = tuple(pre_row[col] for col in pred_pre_cols)
            
            # Check for unpaired points (value of -1)
            if row[post_id_col] == -1:
                post_point = None
            else:
                # Find the post-synaptic point using ID if available, otherwise use index
                if pred_post_id:
                    try:
                        post_row = pred_post_df[pred_post_df[pred_post_id] == row[post_id_col]].iloc[0]
                    except Exception as e:
                        print(f"Warning: Could not find post_id {row[post_id_col]} in pred_post_df")
                        continue
                else:
                    try:
                        post_row = pred_post_df.iloc[row[post_id_col]]
                    except Exception as e:
                        print(f"Warning: Invalid post_id index {row[post_id_col]}")
                        continue
                
                post_point = tuple(post_row[col] for col in pred_post_cols)
            
            pred_pairs.append((pre_point, post_point))
        # Remove None values from pred_pairs
        pred_pairs = [pair for pair in pred_pairs if pair[1] is not None]
    else:
        pred_pairs = [
            (
                tuple(row_pre[1][col] for col in pred_pre_cols),
                tuple(row_post[1][col] for col in pred_post_cols)
            )
            for row_pre, row_post in zip(pred_pre_df.iterrows(), pred_post_df.iterrows())
        ]

    gt_pairs = [
        (
            tuple(row_pre[1][col] for col in gt_pre_cols),
            tuple(row_post[1][col] for col in gt_post_cols)
        )
        for row_pre, row_post in zip(gt_pre_df.iterrows(), gt_post_df.iterrows())
    ]

    print(f"Number of gt_pairs: {len(gt_pairs)}")

    # Calculate F1 score
    fscore, precision, recall, fp, fn, matches = synaptic_partners_fscore(
        pred_pairs, gt_pairs, matching_threshold=args.matching_threshold, all_stats=True
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

    # out_dir = os.path.join(data_path, "results", model_name)
    # os.makedirs(out_dir, exist_ok=True)  #  add a results dir if it is missing
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
