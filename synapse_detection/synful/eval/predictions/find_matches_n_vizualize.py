#! /home/samia/anaconda3/envs/nglancer/bin/python
# conda deactivate; conda activate nglancer : Only works from the terminal and not pycharm

import itertools
import json
import os.path
import sys
import logging
from typing import List, Tuple, Dict

import daisy
import neuroglancer
import numpy as np
import pandas as pd
from funlib.show.neuroglancer import add_layer, ScalePyramid
from synful import database, synapse
from funlib.persistence import open_ds
import h5py
from scipy.spatial import KDTree
import neuroglancer
from collections import defaultdict
import scipy

# neuroglancer.set_server_bind_address('0.0.0.0')
ngid = itertools.count(start=1)


# def crop_synapses_to_roi(synapses: List, roi: List[slice]) -> List:
#     """
#     Crop synapses to specified ROI.
#     """
#     cropped_synapses = []
#
#     for syn in synapses:
#         pre_loc = syn.location_pre
#         if (roi[0].start <= pre_loc[0] <= roi[0].stop and
#                 roi[1].start <= pre_loc[1] <= roi[1].stop and
#                 roi[2].start <= pre_loc[2] <= roi[2].stop):
#             cropped_synapses.append(syn)
#
#     return cropped_synapses


def load_hdf5(inputfilename, dataset):
    f = h5py.File(inputfilename, 'r')
    offset = (0, 0, 0)
    if dataset in f:
        data = f[dataset][:]
        if 'offset' in f[dataset].attrs.keys():
            offset = f[dataset].attrs['offset']
        print(dataset, data.shape, data.dtype)
    else:
        data = None
        print(dataset, 'does not exist')
    f.close()
    return data, offset


def add_synapses(s, directory, source_roi, score_thr=0, radius=30, roi=None):
    synapses = synapse.read_synapses_in_roi(directory, source_roi)
    print(synapses)

    scores = np.array([x.score for x in synapses])
    print(f"score stats: mean {scores.mean()}, max: {scores.max()}, min: {scores.min()}, median: {np.median(scores)}")
    print("Note scores should generally be above the score threshold passed during extraction!")

    # If taking an informed choice: you can use the median score as threshold, uncomment the below for use!
    # score_thr = np.median(scores)
    dir_path = os.path.dirname(directory)

    with open(f"{os.path.join(dir_path, 'score_stats.txt')}", "w") as f:
        f.write(f"Mean: {scores.mean()}\n")
        f.write(f"Max: {scores.max()}\n")
        f.write(f"Min: {scores.min()}\n")
        f.write(f"Median: {np.median(scores)}\n")
        f.write(f"Mode: {scipy.stats.mode(scores)}\n")
        f.write(f"Note scores should generally be above the score threshold passed during extraction!\n")

    # score stats mode
    uniq_scores, counts_unique = np.unique(scores.astype(int), return_counts=True)
    pd.DataFrame({"scores": uniq_scores, "counts": counts_unique}).to_csv(f"{dir_path}/scores_mode.csv", index=False)


    pre_sites = []
    post_sites = []
    connectors = []
    pre_post_mapping = []  # list of tuples: (pre_id, post_id)
    below_score = 0

    pred_pre_sites_locs = []
    pred_post_sites_locs = []
    for syn in synapses:
        if syn.score < score_thr:
            below_score += 1
        else:
            # pre_site = np.flip(syn.location_pre)
            # post_site = np.flip(syn.location_post)

            pre_site = syn.location_pre
            post_site = syn.location_post
            pre_id = next(ngid)

            if roi is not None:  # in zyx
                if (roi[0].start <= pre_site[0] <= roi[0].stop and
                        roi[1].start <= pre_site[1] <= roi[1].stop and
                        roi[2].start <= pre_site[2] <= roi[2].stop):
                    pred_pre_sites_locs.append((pre_id, pre_site))
                    pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
                                                                      radii=(
                                                                          radius, radius, radius),
                                                                      id=id))
            post_id = next(ngid)

            if roi is not None:  # in zyx
                if (roi[0].start <= post_site[0] <= roi[0].stop and
                        roi[1].start <= post_site[1] <= roi[1].stop and
                        roi[2].start <= post_site[2] <= roi[2].stop):
                    pred_post_sites_locs.append((post_id, post_site))
                    post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                                       radii=(
                                                                           radius, radius, radius),
                                                                       id=id))

            connectors.append(
                neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                            id=next(ngid)))
            if roi is not None:
                pre_post_mapping.append((pre_id, post_id))

    # Convert to DataFrame
    df_pre = pd.DataFrame(pred_pre_sites_locs, columns=["Pre_ID", "Pre_Location"])
    df_pre[['Pre_Z', 'Pre_Y', 'Pre_X']] = pd.DataFrame(df_pre["Pre_Location"].tolist(), index=df_pre.index)
    df_pre.drop(columns=["Pre_Location"], inplace=True)

    df_post = pd.DataFrame(pred_post_sites_locs, columns=["Post_ID", "Post_Location"])
    df_post[['Post_Z', 'Post_Y', 'Post_X']] = pd.DataFrame(df_post["Post_Location"].tolist(), index=df_post.index)
    df_post.drop(columns=["Post_Location"], inplace=True)
    df_mapping = pd.DataFrame(pre_post_mapping, columns=['Pre_ID', 'Post_ID'])

    # save the dfs
    df_pre.to_csv(f"{dir_path}/pred_pre_locations.csv")
    df_post.to_csv(f"{dir_path}/pred_post_locations.csv")
    df_mapping.to_csv(f"{dir_path}/pre_post_mapping.csv")

    s.layers.append(
        name="connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1]  # [1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            # linked_segmentation_layer={'connectors': 'segmentation'},
            # filter_by_segmentation=['connectors'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ffff00',
                )
            ],
            annotations=connectors
        )
    )

    s.layers.append(
        name="pre_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1]  # [1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            linked_segmentation_layer={'pre_sites': 'segmentation'},
            filter_by_segmentation=['pre_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='red'  # '#ff0000',  # '#ffff00',
                )
            ],
            annotations=pre_sites
        )
    )

    s.layers.append(
        name="post_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1]  # [1, 1, 1],
            ),
            annotation_relationships=['connectors'],
            # linked_segmentation_layer={'post_sites': 'segmentation'},
            # filter_by_segmentation=['post_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ff00ff'  # '#ff00ff',
                )
            ],
            annotations=post_sites
        )
    )

    print(
        'filtered out {}/{} of synapses'.format(below_score,
                                                len(synapses)))
    print('displaying {} synapses'.format(len(post_sites)))

    return pred_pre_sites_locs, pred_post_sites_locs


def add_cremi_synapses(s, filename, res=(8, 8, 8), roi=None):
    if filename.lower().endswith((".h5", ".hdf", ".hdf5")):
        # inelegant loading for now till open_ds gets fixed for hdfs (loads zarrs ok)
        locs, offsets = load_hdf5(filename, 'annotations/locations')
    elif filename.lower().endswith(".zarr"):
        locs, offsets = load_zarr(filename, 'annotations/locations')

    offset = 0

    dir_path = os.path.dirname(filename)

    # convert offsets to voxel coords
    offsets = [i / j for i, j in zip(offsets, res)]

    # flip locations if data gets loaded as xyz
    # locs = [np.flip(loc) + offset for loc in locs]
    # currently data is loaded as zyx
    locs = [loc + np.array(offsets, dtype=np.float32) for loc in locs]
    # locs = [loc + offset for loc in locs]
    print(locs)

    if filename.lower().endswith((".h5", ".hdf", ".hdf5")):
        partners, offset = load_hdf5(filename, 'annotations/presynaptic_site/partners')
        annotation_ids, offset = load_hdf5(filename, 'annotations/ids')
    elif filename.lower().endswith(".zarr"):
        partners, offset = load_zarr(filename, 'annotations/presynaptic_site/partners')
        annotation_ids, offset = load_zarr(filename, 'annotations/ids')

    (pre_sites, post_sites, connectors) = ([], [], [])
    (gt_pre_sites_loc, gt_post_sites_loc, gt_connectors_loc) = ([], [], [])
    pre_post_mapping = []  # list of tuples: (pre_id, post_id)
    for (pre, post) in partners:

        # Get indices of rows where pre matches the first column of annotation_ids
        pre_indices = np.where(annotation_ids == pre)[0]
        # Get indices of rows where post matches the second column of annotation_ids
        post_indices = np.where(annotation_ids == post)[0]

        # pre_index = int(np.where(pre == annotation_ids)[0][0])
        # post_index = int(np.where(post == annotation_ids)[0][0])

        # pre_indices = np.where(pre == annotation_ids)[0]
        # post_indices = np.where(post == annotation_ids)[0]

        if len(pre_indices) == 0 or len(post_indices) == 0:
            print(f"Skipping pair (pre: {pre}, post: {post}) - not found in annotation_ids")
            continue

        pre_index = int(pre_indices[0])
        post_index = int(post_indices[0])

        # print(pre_index, post_index)
        pre_site = locs[pre_index]
        post_site = locs[post_index]

        # get rid of the transpose if the data is already transposed
        # post_site = [post_site[2], post_site[1], post_site[0]]
        # pre_site = [pre_site[2], pre_site[1], pre_site[0]]

        # pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
        #                                                   radii=(40, 40, 40),
        #                                                   id=next(ngid)))

        post_id = next(ngid)

        if roi is not None:  # in zyx
            if (roi[0].start <= post_site[0] <= roi[0].stop and
                    roi[1].start <= post_site[1] <= roi[1].stop and
                    roi[2].start <= post_site[2] <= roi[2].stop):
                gt_post_sites_loc.append((post_id, post_site))
                post_sites.append(neuroglancer.PointAnnotation(point=post_site,
                                                               # radii=(10, 10, 10),
                                                               id=id))

        pre_id = next(ngid)

        if roi is not None:  # in zyx
            # print(
            # f"roi 0 start-stop {roi[0].start, roi[0].stop}, 1 {roi[1].start, roi[1].stop}, "
            # f"2 {roi[2].start, roi[2].stop}, pre-site {pre_site} ")
            if (roi[0].start <= pre_site[0] <= roi[0].stop and
                    roi[1].start <= pre_site[1] <= roi[1].stop and
                    roi[2].start <= pre_site[2] <= roi[2].stop):
                gt_pre_sites_loc.append((pre_id, pre_site))
                pre_sites.append(neuroglancer.PointAnnotation(point=pre_site,
                                                              # radii=(10, 10, 10),
                                                              id=id))

        # post_sites.append(neuroglancer.PointAnnotation(point=(100, 100, 100),
        #                                                    # radii=(40, 40, 40),
        #                                                    id=next(ngid)))
        connectors.append(
            neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                        id=next(ngid)
                                        ))
        if roi is not None:
            pre_post_mapping.append((pre_id, post_id))

    # Convert to DataFrame
    df_pre = pd.DataFrame(gt_pre_sites_loc, columns=["Pre_ID", "Pre_Location"])
    df_pre[['Pre_Z', 'Pre_Y', 'Pre_X']] = pd.DataFrame(df_pre["Pre_Location"].tolist(), index=df_pre.index)
    df_pre.drop(columns=["Pre_Location"], inplace=True)

    df_post = pd.DataFrame(gt_post_sites_loc, columns=["Post_ID", "Post_Location"])
    df_post[['Post_Z', 'Post_Y', 'Post_X']] = pd.DataFrame(df_post["Post_Location"].tolist(), index=df_post.index)
    df_post.drop(columns=["Post_Location"], inplace=True)
    df_mapping = pd.DataFrame(pre_post_mapping, columns=['Pre_ID', 'Post_ID'])

    # save the dfs
    df_pre.to_csv(f"{dir_path}/{os.path.basename(filename).split('.')[0]}_gt_pre_locations.csv")
    df_post.to_csv(f"{dir_path}/{os.path.basename(filename).split('.')[0]}_gt_post_locations.csv")
    df_mapping.to_csv(f"{dir_path}/{os.path.basename(filename).split('.')[0]}_gt_pre_post_mapping.csv")

    # print(f"Connectors: {connectors}")
    # print(f"pre_sites: {pre_sites}")
    # print(f"post sites: {post_sites}")

    s.layers.append(
        name="gt_connectors",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1],  # [8, 8, 8]
            ),
            annotation_relationships=['gt_connectors'],
            # linked_segmentation_layer={'connectors': 'segmentation'},
            # filter_by_segmentation=['connectors'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ffff00',
                )
            ],
            annotations=connectors
        )
    )

    s.layers.append(
        name="gt_pre_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1],  # [8, 8, 8]
            ),
            annotation_relationships=['gt_connectors'],
            linked_segmentation_layer={'gt_pre_sites': 'segmentation'},
            filter_by_segmentation=['pre_sites'],
            ignore_null_segment_filter=False,
            annotation_color='#ff0000',
            # annotation_properties=[
            #     neuroglancer.AnnotationPropertySpec(
            #         id='color',
            #         type='rgb',
            #         default='#FF0000'  # '#ff0000',  # '#ffff00',
            #     )
            # ],
            annotations=pre_sites
        )
    )

    s.layers.append(
        name="gt_post_sites",
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                # names=["x", "y", "z"],
                units="nm",
                scales=[1, 1, 1],  # [8, 8, 8]
            ),
            annotation_relationships=['gt_connectors'],
            # linked_segmentation_layer={'post_sites': 'segmentation'},
            # filter_by_segmentation=['post_sites'],
            ignore_null_segment_filter=False,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id='color',
                    type='rgb',
                    default='#ff00ff'  # '#ff00ff',
                )
            ],
            annotations=post_sites
        )
    )

    return gt_pre_sites_loc, gt_post_sites_loc


# def add_cremi_synapses(s, filename):
#     offset = 0
#     locs = open_ds(filename, 'annotations/locations')
#
#     locs = [np.flip(loc) + offset for loc in locs.data]
#     partners = open_ds(filename, 'annotations/presynaptic_site/partners')
#     annotation_ids = open_ds(filename, 'annotations/ids').data
#
#     (pre_sites, post_sites, connectors) = ([], [], [])
#     distances = []
#     for (pre, post) in partners.data:
#         pre_index = int(np.where(pre == annotation_ids)[0][0])
#         post_index = int(np.where(post == annotation_ids)[0][0])
#         pre_site = locs[pre_index]
#         post_site = locs[post_index]
#
#         pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
#                                                           radii=(40, 40, 40),
#                                                           id=next(ngid)))
#         post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
#                                                            radii=(40, 40, 40),
#                                                            id=next(ngid)))
#         connectors.append(
#             neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
#                                         id=next(ngid)))
#         dist = np.linalg.norm(np.array(list(pre_site)) - np.array(list(post_site)))
#         distances.append(dist)
#     print(np.mean(distances), np.median(distances))
#     print(len(distances))
#     s.layers['connetors_gt'] = neuroglancer.AnnotationLayer(
#         voxel_size=(1, 1, 1),
#         filter_by_segmentation=False,
#         annotation_color='#ffff00',
#         annotations=connectors,
#     )
#     s.layers['pre_sites_gt'] = neuroglancer.AnnotationLayer(
#         voxel_size=(1, 1, 1),
#         filter_by_segmentation=False,
#         annotation_color='#00ff00',
#         annotations=pre_sites,
#     )
#     s.layers['post_sites_gt'] = neuroglancer.AnnotationLayer(
#         voxel_size=(1, 1, 1),
#         filter_by_segmentation=False,
#         annotation_color='#ff00ff',
#         annotations=post_sites,
#     )


def match_pre_locations_with_dedup(pred_pre_synapses: List[Tuple],
                                   gt_pre_synapses: List[Tuple],
                                   distance_threshold: float = 100.0) -> Dict:
    """
    Match predicted pre-synaptic locations to ground truth using KDTree, with GT deduplication.

    Args:
        pred_pre_synapses: List of tuples (id, location array) for predicted pre-synaptic sites.
        gt_pre_synapses: List of tuples (id, location array) for ground truth pre-synaptic sites.
        distance_threshold: Maximum distance for a match in the same units as coordinates.

    Returns:
        Dictionary containing matching information and statistics.
    """
    # Extract predicted IDs and locations
    pred_ids = [p[0] for p in pred_pre_synapses]
    pred_locations = np.array([p[1] for p in pred_pre_synapses])

    # Extract GT IDs and locations
    gt_ids = [g[0] for g in gt_pre_synapses]
    gt_locations = np.array([g[1] for g in gt_pre_synapses])

    # Deduplicate GT locations
    _, unique_indices = np.unique(gt_locations, axis=0, return_index=True)
    gt_ids = [gt_ids[i] for i in unique_indices]
    gt_locations = gt_locations[unique_indices]

    print(f"Deduplicated GT locations: {len(gt_locations)} unique points")

    # Build KDTree for deduplicated GT locations
    tree = KDTree(gt_locations)

    # Find nearest neighbors and distances
    distances, indices = tree.query(pred_locations, k=1)

    # Filter matches based on distance threshold
    valid_matches = distances <= distance_threshold

    # Organize matches by GT synapse
    gt_to_predictions = defaultdict(list)
    for i, is_valid in enumerate(valid_matches):
        if is_valid:
            gt_idx = indices[i]
            gt_to_predictions[gt_idx].append((distances[i], pred_ids[i], pred_locations[i]))

    # Apply Non-Maximum Suppression (NMS)
    matched_pairs = []
    for gt_idx, preds in gt_to_predictions.items():
        # Sort predictions for this GT by distance and keep the closest
        preds.sort(key=lambda x: x[0])  # Sort by distance
        best_pred = preds[0]  # Closest prediction
        matched_pairs.append({
            'pred_id': best_pred[1],
            'pred_loc': best_pred[2],
            'gt_id': gt_ids[gt_idx],
            'gt_loc': gt_locations[gt_idx],
            'distance': best_pred[0]
        })

    # Sort matches by distance
    matched_pairs.sort(key=lambda x: x['distance'])

    # Identify unmatched predictions and ground truth
    matched_pred_ids = {m['pred_id'] for m in matched_pairs}
    matched_gt_ids = {m['gt_id'] for m in matched_pairs}

    unmatched_pred = [(pred_ids[i], pred_locations[i])
                      for i in range(len(pred_locations))
                      if pred_ids[i] not in matched_pred_ids]

    unmatched_gt = [(gt_ids[i], gt_locations[i])
                    for i in range(len(gt_locations))
                    if gt_ids[i] not in matched_gt_ids]

    # Compile results
    results = {
        'matched_pairs': matched_pairs,
        'unmatched_pred': unmatched_pred,
        'unmatched_gt': unmatched_gt,
        'stats': {
            'total_pred': len(pred_locations),
            'total_gt': len(gt_locations),
            'num_matches': len(matched_pairs),
            'precision': len(matched_pairs) / len(pred_locations) if len(pred_locations) > 0 else 0,
            'recall': len(matched_pairs) / len(gt_locations) if len(gt_locations) > 0 else 0
        }
    }

    return results


def match_pre_locations(pred_pre_synapses: List[Tuple],
                        gt_pre_synapses: List[Tuple],
                        distance_threshold: float = 200.0) -> Dict:
    """
    Match predicted pre-synaptic locations to ground truth using KDTree.

    Args:
        pred_pre_synapses: List of tuples (id, location array) for predicted pre-synaptic sites
        gt_pre_synapses: List of tuples (id, location array) for ground truth pre-synaptic sites
        distance_threshold: Maximum distance for a match in the same units as coordinates

    Returns:
        Dictionary containing matching information and statistics
    """
    # Extract locations and IDs
    pred_ids = [p[0] for p in pred_pre_synapses]
    pred_locations = np.array([p[1] for p in pred_pre_synapses])

    gt_ids = [g[0] for g in gt_pre_synapses]
    gt_locations = np.array([g[1] for g in gt_pre_synapses])

    # Build KDTree for ground truth locations
    tree = KDTree(gt_locations)

    # Find nearest neighbors and distances
    distances, indices = tree.query(pred_locations, k=1)
    # Unique indices
    u_indices = np.unique(indices)
    print(f"unique indices {u_indices}, len: {len(u_indices)}")

    # Filter matches based on distance threshold
    valid_matches = distances <= distance_threshold

    # # Create list of matched pairs with their distances
    # matched_pairs = []
    # for i in range(len(pred_locations)):
    #     if valid_matches[i]:
    #         matched_pairs.append({
    #             'pred_id': pred_ids[i],
    #             'pred_loc': pred_locations[i],
    #             'gt_id': gt_ids[indices[i]],
    #             'gt_loc': gt_locations[indices[i]],
    #             'distance': distances[i]
    #         })

    # Organize matches by GT synapse
    gt_to_predictions = defaultdict(list)
    for i, is_valid in enumerate(valid_matches):
        if is_valid:
            gt_idx = indices[i]
            gt_to_predictions[gt_idx].append((distances[i], pred_ids[i], pred_locations[i]))

    # Apply Non-Maximum Suppression (NMS)
    matched_pairs = []
    for gt_idx, preds in gt_to_predictions.items():
        # Sort predictions for this GT by distance and keep the closest
        preds.sort(key=lambda x: x[0])  # Sort by distance
        best_pred = preds[0]  # Closest prediction
        matched_pairs.append({
            'pred_id': best_pred[1],
            'pred_loc': best_pred[2],
            'gt_id': gt_ids[gt_idx],
            'gt_loc': gt_locations[gt_idx],
            'distance': best_pred[0]
        })
    # Sort matches by distance
    matched_pairs.sort(key=lambda x: x['distance'])

    # # Get unmatched predictions and ground truth
    # matched_pred_indices = set([i for i, v in enumerate(valid_matches) if v])
    # matched_gt_indices = set(indices[valid_matches])
    #
    # unmatched_pred = [(pred_ids[i], pred_locations[i])
    #                   for i in range(len(pred_locations))
    #                   if i not in matched_pred_indices]
    #
    # unmatched_gt = [(gt_ids[i], gt_locations[i])
    #                 for i in range(len(gt_locations))
    #                 if i not in matched_gt_indices]
    #
    # # Compile results
    # results = {
    #     'matched_pairs': matched_pairs,
    #     'unmatched_pred': unmatched_pred,
    #     'unmatched_gt': unmatched_gt,
    #     'stats': {
    #         'total_pred': len(pred_locations),
    #         'total_gt': len(gt_locations),
    #         'num_matches': len(matched_pairs),
    #         'precision': len(matched_pairs) / len(pred_locations) if len(pred_locations) > 0 else 0,
    #         'recall': len(matched_pairs) / len(gt_locations) if len(gt_locations) > 0 else 0
    #     }
    # }

    # Identify unmatched predictions and ground truth
    matched_pred_ids = {m['pred_id'] for m in matched_pairs}
    matched_gt_ids = {m['gt_id'] for m in matched_pairs}

    unmatched_pred = [(pred_ids[i], pred_locations[i])
                      for i in range(len(pred_locations))
                      if pred_ids[i] not in matched_pred_ids]

    unmatched_gt = [(gt_ids[i], gt_locations[i])
                    for i in range(len(gt_locations))
                    if gt_ids[i] not in matched_gt_ids]

    # Compile results
    results = {
        'matched_pairs': matched_pairs,
        'unmatched_pred': unmatched_pred,
        'unmatched_gt': unmatched_gt,
        'stats': {
            'total_pred': len(pred_locations),
            'total_gt': len(gt_locations),
            'num_matches': len(matched_pairs),
            'precision': len(matched_pairs) / len(pred_locations) if len(pred_locations) > 0 else 0,
            'recall': len(matched_pairs) / len(gt_locations) if len(gt_locations) > 0 else 0
        }
    }

    return results


def match_pre_post_locations(pred_pre_synapses: List[Tuple],
                             pred_post_synapses: List[Tuple],
                             gt_pre_synapses: List[Tuple],
                             gt_post_synapses: List[Tuple],
                             distance_threshold: float = 200.0) -> Dict:
    """
    Match predicted pre- and post-synaptic pairs to ground truth using KDTree.

    Args:
        pred_pre_synapses: List of tuples (id, location array) for predicted pre-synaptic sites.
        pred_post_synapses: List of tuples (id, location array) for predicted post-synaptic sites.
        gt_pre_synapses: List of tuples (id, location array) for ground truth pre-synaptic sites.
        gt_post_synapses: List of tuples (id, location array) for ground truth post-synaptic sites.
        distance_threshold: Maximum distance for a match in the same units as coordinates.

    Returns:
        Dictionary containing matching information and statistics.
    """
    # Extract IDs and locations
    pred_pre_ids = [p[0] for p in pred_pre_synapses]
    pred_pre_locations = np.array([p[1] for p in pred_pre_synapses])
    pred_post_ids = [p[0] for p in pred_post_synapses]
    pred_post_locations = np.array([p[1] for p in pred_post_synapses])

    gt_pre_ids = [g[0] for g in gt_pre_synapses]
    gt_pre_locations = np.array([g[1] for g in gt_pre_synapses])
    gt_post_ids = [g[0] for g in gt_post_synapses]
    gt_post_locations = np.array([g[1] for g in gt_post_synapses])

    # Deduplicate GT pre-synaptic locations
    _, unique_pre_indices = np.unique(gt_pre_locations, axis=0, return_index=True)
    gt_pre_ids = [gt_pre_ids[i] for i in unique_pre_indices]
    gt_pre_locations = gt_pre_locations[unique_pre_indices]

    # Build KDTree for pre- and post-synaptic locations
    pre_tree = KDTree(gt_pre_locations)
    post_tree = KDTree(gt_post_locations)

    # Find nearest neighbors and distances for pre- and post-synaptic sites
    pre_distances, pre_indices = pre_tree.query(pred_pre_locations, k=1)
    post_distances, post_indices = post_tree.query(pred_post_locations, k=1)

    # Filter matches based on distance threshold
    valid_pre_matches = pre_distances <= distance_threshold
    valid_post_matches = post_distances <= distance_threshold

    # Combine pre and post matches to form pairs
    matched_pairs = []
    for i in range(len(pred_pre_locations)):
        if valid_pre_matches[i] and valid_post_matches[i]:
            gt_pre_idx = pre_indices[i]
            gt_post_idx = post_indices[i]
            gt_pre_id = gt_pre_ids[gt_pre_idx]
            gt_post_id = gt_post_ids[gt_post_idx]
            matched_pairs.append({
                'pred_pre_id': pred_pre_ids[i],
                'pred_pre_loc': pred_pre_locations[i],
                'pred_post_id': pred_post_ids[i],
                'pred_post_loc': pred_post_locations[i],
                'gt_pre_id': gt_pre_id,
                'gt_pre_loc': gt_pre_locations[gt_pre_idx],
                'gt_post_id': gt_post_id,
                'gt_post_loc': gt_post_locations[gt_post_idx],
                'pre_distance': pre_distances[i],
                'post_distance': post_distances[i]
            })

    # Identify unique GT pre-post pairs
    unique_gt_pairs = set(zip(gt_pre_ids, gt_post_ids))

    # Identify unmatched predictions and ground truth
    matched_pred_ids = {(m['pred_pre_id'], m['pred_post_id']) for m in matched_pairs}
    unmatched_pred = [(pred_pre_ids[i], pred_pre_locations[i], pred_post_ids[i], pred_post_locations[i])
                      for i in range(len(pred_pre_locations))
                      if (pred_pre_ids[i], pred_post_ids[i]) not in matched_pred_ids]

    unmatched_gt = [(gt_pre_ids[i], gt_pre_locations[i], gt_post_ids[i], gt_post_locations[i])
                    for i in range(len(gt_pre_ids))
                    if (gt_pre_ids[i], gt_post_ids[i]) not in unique_gt_pairs]

    # Compile results
    results = {
        'matched_pairs': matched_pairs,
        'unmatched_pred': unmatched_pred,
        'unmatched_gt': unmatched_gt,
        'stats': {
            'total_pred': len(pred_pre_locations),
            'total_gt': len(gt_pre_locations),
            'num_matches': len(matched_pairs),
            'precision': len(matched_pairs) / len(pred_pre_locations) if len(pred_pre_locations) > 0 else 0,
            'recall': len(matched_pairs) / len(unique_gt_pairs) if len(unique_gt_pairs) > 0 else 0
        }
    }

    return results


def check_location_overlaps(matched_pairs: List[Dict],
                            unmatched_pred: List[Tuple],
                            unmatched_gt: List[Tuple],
                            distance_threshold: float = 100.0) -> Dict:
    """
    Check for spatial overlaps between matched and unmatched locations.

    Args:
        matched_pairs: List of dictionaries containing matched pairs
        unmatched_pred: List of (id, location) tuples for unmatched predictions
        unmatched_gt: List of (id, location) tuples for unmatched ground truth
        distance_threshold: Distance threshold to consider as overlap
    """
    # Extract locations from matched pairs
    matched_pred_locs = np.array([m['pred_loc'] for m in matched_pairs])
    matched_gt_locs = np.array([m['gt_loc'] for m in matched_pairs])

    # Extract unmatched locations
    unmatched_pred_locs = np.array([loc for _, loc in unmatched_pred])
    unmatched_gt_locs = np.array([loc for _, loc in unmatched_gt])

    overlaps = {
        'matched_pred_vs_unmatched_pred': [],
        'matched_gt_vs_unmatched_gt': [],
        'unmatched_pred_vs_unmatched_gt': []
    }

    # Check matched pred vs unmatched pred
    if len(matched_pred_locs) > 0 and len(unmatched_pred_locs) > 0:
        tree = KDTree(matched_pred_locs)
        distances, indices = tree.query(unmatched_pred_locs)
        overlaps['matched_pred_vs_unmatched_pred'] = [
            (unmatched_pred[i][0], matched_pairs[idx]['pred_id'], dist)
            for i, (dist, idx) in enumerate(zip(distances, indices))
            if dist <= distance_threshold
        ]

    # Check matched gt vs unmatched gt
    if len(matched_gt_locs) > 0 and len(unmatched_gt_locs) > 0:
        tree = KDTree(matched_gt_locs)
        distances, indices = tree.query(unmatched_gt_locs)
        overlaps['matched_gt_vs_unmatched_gt'] = [
            (unmatched_gt[i][0], matched_pairs[idx]['gt_id'], dist)
            for i, (dist, idx) in enumerate(zip(distances, indices))
            if dist <= distance_threshold
        ]

    # Check unmatched pred vs unmatched gt
    if len(unmatched_pred_locs) > 0 and len(unmatched_gt_locs) > 0:
        tree = KDTree(unmatched_pred_locs)
        distances, indices = tree.query(unmatched_gt_locs)
        overlaps['unmatched_pred_vs_unmatched_gt'] = [
            (unmatched_gt[i][0], unmatched_pred[indices[i]][0], dist)
            for i, dist in enumerate(distances)
            if dist <= distance_threshold
        ]

    return overlaps


def create_multi_view_visualization(matches: Dict, raw, synapsedir, roi, gt_synfile, viewer=None, s=None):
    """
    Create a Neuroglancer viewer with multiple views for different synapse categories.
    """
    if viewer is None:
        viewer = neuroglancer.Viewer()

    # Create shared coordinate space
    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units="nm",
        scales=[1, 1, 1]
    )

    # Annotation sources
    matched_pairs_annotations = [
        neuroglancer.LineAnnotation(
            point_a=m['pred_loc'],
            point_b=m['gt_loc'],
            id=str(i)
        ) for i, m in enumerate(matches['matched_pairs'])
    ]

    unmatched_pred_annotations = [
        neuroglancer.PointAnnotation(
            point=loc,
            id=f"pred_{id_}"
        ) for id_, loc in matches['unmatched_pred']
    ]

    unmatched_gt_annotations = [
        neuroglancer.PointAnnotation(
            point=loc,
            id=f"gt_{id_}"
        ) for id_, loc in matches['unmatched_gt']
    ]

    if viewer is None:
        with viewer.txn() as s:
            # Viewer transaction
            s.dimensions = dimensions

            # Matched pairs layer
            s.layers['matched_pairs'] = neuroglancer.AnnotationLayer(
                annotations=matched_pairs_annotations,
                annotation_color='#00ff00'
            )

            # Unmatched predictions layer
            s.layers['unmatched_predictions'] = neuroglancer.AnnotationLayer(
                annotations=unmatched_pred_annotations,
                annotation_color='#ff0000'
            )

            # Unmatched ground truth layer
            s.layers['unmatched_ground_truth'] = neuroglancer.AnnotationLayer(
                annotations=unmatched_gt_annotations,
                annotation_color='#0000ff'
            )

            # Overview layer with all annotations
            s.layers['overview'] = neuroglancer.AnnotationLayer(
                annotations=(
                        matched_pairs_annotations +
                        unmatched_pred_annotations +
                        unmatched_gt_annotations
                ),
                annotation_color='#ff00ff'
            )

            # Define the layout with rows and columns
            s.layout = neuroglancer.row_layout([
                neuroglancer.column_layout([
                    neuroglancer.LayerGroupViewer(layers=['matched_pairs']),
                    neuroglancer.LayerGroupViewer(layers=['unmatched_predictions']),
                ]),
                neuroglancer.column_layout([
                    neuroglancer.LayerGroupViewer(layers=['unmatched_ground_truth']),
                    neuroglancer.LayerGroupViewer(layers=['overview']),
                ]),
            ])

            # Link all views
            s.cross_section_scale = 1.0
            s.position = matches['matched_pairs'][0]['pred_loc'] if matches['matched_pairs'] else \
                matches['unmatched_pred'][0][1]
    else:
        # Viewer transaction
        s.dimensions = dimensions

        # Matched pairs layer
        s.layers['matched_pairs'] = neuroglancer.AnnotationLayer(
            annotations=matched_pairs_annotations,
            annotation_color='#00ff00'
        )

        # Unmatched predictions layer
        s.layers['unmatched_predictions'] = neuroglancer.AnnotationLayer(
            annotations=unmatched_pred_annotations,
            annotation_color='#ff0000'
        )

        # Unmatched ground truth layer
        s.layers['unmatched_ground_truth'] = neuroglancer.AnnotationLayer(
            annotations=unmatched_gt_annotations,
            annotation_color='#0000ff'
        )

        # Overview layer with all annotations
        s.layers['overview'] = neuroglancer.AnnotationLayer(
            annotations=(
                    matched_pairs_annotations +
                    unmatched_pred_annotations +
                    unmatched_gt_annotations
            ),
            annotation_color='#ff00ff'
        )

        # Define the layout with rows and columns
        # s.layout = neuroglancer.row_layout([
        #     neuroglancer.column_layout([
        #         neuroglancer.LayerGroupViewer(layers=['matched_pairs']),
        #         neuroglancer.LayerGroupViewer(layers=['unmatched_predictions']),
        #     ]),
        #     neuroglancer.column_layout([
        #         neuroglancer.LayerGroupViewer(layers=['unmatched_ground_truth']),
        #         neuroglancer.LayerGroupViewer(layers=['overview']),
        #     ]),
        # ])

        # Link all views
        # s.cross_section_scale = 1.0
        # s.position = matches['matched_pairs'][0]['pred_loc'] if matches['matched_pairs'] else \
        #     matches['unmatched_pred'][0][1]


def analyze_and_visualize_matches(matches, raw, synapsedir, roi, gt_synfile, viewer=None, s=None):
    # Check for overlaps
    overlaps = check_location_overlaps(
        matches['matched_pairs'],
        matches['unmatched_pred'],
        matches['unmatched_gt']
    )

    # Print overlap analysis
    # print("\nOverlap Analysis:")
    # print("\nMatched predictions vs Unmatched predictions:")
    # for pred_id, matched_id, dist in overlaps['matched_pred_vs_unmatched_pred']:
    #     print(f"Unmatched pred {pred_id} overlaps with matched pred {matched_id} (distance: {dist:.2f})")
    #
    # print("\nMatched ground truth vs Unmatched ground truth:")
    # for gt_id, matched_id, dist in overlaps['matched_gt_vs_unmatched_gt']:
    #     print(f"Unmatched GT {gt_id} overlaps with matched GT {matched_id} (distance: {dist:.2f})")
    #
    # print("\nUnmatched predictions vs Unmatched ground truth:")
    # for gt_id, pred_id, dist in overlaps['unmatched_pred_vs_unmatched_gt']:
    #     print(f"Unmatched GT {gt_id} is close to unmatched pred {pred_id} (distance: {dist:.2f})")

    # Create visualization
    create_multi_view_visualization(matches, raw, synapsedir, roi, gt_synfile, viewer=viewer, s=s)
    # print(f"\nNeuroglancer URL: {viewer}")

    return overlaps, viewer


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # gt roi coords in zyx?? Calculated from a difference in coords in px space of the bboxes used
    octo_cube1_roi = None  # [slice(0 * 8, 679 * 8, None), slice(0 * 8, 670 * 8, None), slice(0 * 8, 669 * 8, None)]
    voxel_resolution = (8, 8, 8)

    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/OCTO_3CUBES_ZYX/data_3d/train/octo_cube1_8083_8765_y5878_6542_z4697_5319.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/OCTO_3CUBES_ZYX/data_3d/train/octo_cube2_12485_13164_y6231_6901_z3971_4640.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/OCTO_3CUBES_ZYX/data_3d/train/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/HEMI_REFINED_SYN/data_3d/train/synapses_x12437-13037_y27229-27829_z17176-17776.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/HEMI_REFINED_SYN/data_3d/train/synapses_x15082-15682_y31050-31650_z14555-15155.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/MR143/data_3d/test/mr143_cube1_8190_8690_y6896_7396_z12790_13290.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/MR143/data_3d/test/mr143_cube2_8633_9133_y6069_6569_z9225_9725.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/PARKER/data_3d/test/parker_cube1_16852_17620_y7286_8054_z1506_2274.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/PARKER/data_3d/test/parker_cube2_6512_7280_y7774_8542_z1619_2387.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/hemi_synapses_x15035-15635_y28559-29159_z9602-10202.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/manc_synapses_x23400-24000_y24000-24600_z14400-15000.hdf'
    # trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/wasp_train_vol0_syns_zyx_2217-2617_4038-4448_6335-6735_cremi_same_preid.hdf'
    trainingfile = '/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/test/octo_cube1_8083_8765_y5878_6542_z4697_5319.hdf'
    # trainingfile = "/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/synapses_x12437-13037_y27229-27829_z17176-17776.hdf"

    # NEURIPS TRAINING DATA
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/manc_synapses_x14200-14800_y33000-33600_z44600-45200.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/manc_synapses_x20200-20800_y34200-34800_z32200-32800.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/octo_cube2_12485_13164_y6231_6901_z3971_4640.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/synapses_x12437-13037_y27229-27829_z17176-17776.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/synapses_x15082-15682_y31050-31650_z14555-15155.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/synapses_x21786-22386_y28978-29578_z18787-19387.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/synapses_x27262-27862_y31539-32139_z17577-18177.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/train_vol1_syns_zyx_3680-4096_2944-3360_4448-4864_cremi_same_preid.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/train_vol2_syns_zyx_3776-4192_6048-6464_9248-9664_cremi_same_preid.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/train_vol3_syns_zyx_5152-5568_3168-3584_8384-8800_cremi_same_preid.hdf"
    # trainingfile="/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/train/train_vol4_syns_zyx_1920-2336_4832-5248_6528-6944_cremi_same_preid.hdf"


    # neuron_ds = '/volumes/labels/neuron_ids'
    # mask_ds = 'volumes/masks/groundtruth'
    raw_ds = 'volumes/raw'
    # neuron = daisy.open_ds(trainingfile, neuron_ds)
    # mask = daisy.open_ds(trainingfile, mask_ds)
    raw = open_ds(trainingfile, raw_ds)

    # let's calculate the roi based on input if not give above
    if not octo_cube1_roi:
        input_shape = raw.shape
        octo_cube1_roi = [slice(0, input_shape[0] * voxel_resolution[0], None),
                          slice(0, input_shape[1] * voxel_resolution[1], None),
                          slice(0, input_shape[2] * voxel_resolution[2], None), ]
        print(f"roi {octo_cube1_roi}")
    # inferencefile = '/media/samia/DATA/mou1nts/zstore1/synful/scripts/predict_dec_allcubes/output_predict_on_train/octo/setup_03_octo_cube_all3/325000/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.zarr'
    # pred_post_syn = 'volumes/pred_syn_indicator'
    # pred_post_dir = 'volumes/pred_partner_vectors'
    # pred_post_syn = open_ds(inferencefile, pred_post_syn)
    # pred_post_dir = open_ds(inferencefile, pred_post_dir)

    synapsedir = '/media/samia/DATA/mounts/zstore1/synful/scripts/predict/output_predict_on_train/octo_cube1_sc100_setup_03_neurips_octo_labels_300000/syn_cc_thr095000_sum'
    #
    # gt_synfile = '/media/samia/DATA/mounts/zstore1/catena/data/OCTO_3CUBES_ZYX/data_3d/train/octo_cube1_8083_8765_y5878_6542_z4697_5319.hdf'
    gt_synfile = trainingfile

    # Create a custom shader for direction vectors.
    nmfactor = 1
    clipvalue = 100
    pred_shader = """void main() {{ emitRGB(vec3((
        clamp(getDataValue(0)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
        clamp(getDataValue(1)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
        clamp(getDataValue(2)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f})); }}""".format(
        str(nmfactor), clipvalue, clipvalue * 2)

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        # add_layer(s, neuron, 'neurons')
        # add_layer(s, mask, 'mask')
        add_layer(s, raw, 'raw')
        # add_layer(s, pred_post_syn, 'pred_syn')
        # add_layer(s, pred_post_dir, 'pred_dir', shader=pred_shader)
        gt_pre_synapses, gt_post_synapses = add_cremi_synapses(s, gt_synfile, roi=octo_cube1_roi)

        pred_pre_synapses, pred_post_synapses = add_synapses(s, synapsedir, raw.roi, score_thr=770, roi=octo_cube1_roi)

        # match pre detections with gt
        matches = match_pre_locations_with_dedup(pred_pre_synapses, gt_pre_synapses, distance_threshold=100)
        print(f"Matching Statistics:")
        print(f"Total predicted pre-synaptic sites: {matches['stats']['total_pred']}")
        print(f"Total ground truth pre-synaptic sites: {matches['stats']['total_gt']}")
        print(f"Number of matches found: {matches['stats']['num_matches']}")
        print(f"Precision: {matches['stats']['precision']:.3f}")
        print(f"Recall: {matches['stats']['recall']:.3f}")

        # Print first few matches sorted by distance
        # print("\nFirst 10 matches (sorted by distance):")
        # for i, match in enumerate(matches['matched_pairs'][:10]):
        #     print(f"\nMatch {i + 1}:")
        #     print(f"Distance: {match['distance']:.2f}")
        #     print(f"Predicted ID: {match['pred_id']}, Location: {match['pred_loc']}")
        #     print(f"Ground Truth ID: {match['gt_id']}, Location: {match['gt_loc']}")

        results_pairs = match_pre_post_locations(pred_pre_synapses, pred_post_synapses,
                                                 gt_pre_synapses, gt_post_synapses,
                                                 distance_threshold=100.0)

        print(f"\nPair-Wise Matching Statistics:")
        print(f"Total predicted pre-synaptic sites: {results_pairs['stats']['total_pred']}")
        print(f"Total ground truth pre-synaptic sites: {results_pairs['stats']['total_gt']}")
        print(f"Number of matches found: {results_pairs['stats']['num_matches']}")
        print(f"Precision: {results_pairs['stats']['precision']:.3f}")
        print(f"Recall: {results_pairs['stats']['recall']:.3f}")

        # analyse any overlaps in matched and unmatched pairs
        analyze_and_visualize_matches(matches, raw=raw, synapsedir=synapsedir, roi=octo_cube1_roi,
                                      gt_synfile=gt_synfile, viewer=None, s=s)
    print(f"\nNeuroglancer URL: {viewer.__str__()}")
