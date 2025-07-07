"""
Script that calculates the Precision Recall and F1 independently for the Pre and Post points of the synapse predictions.

Main Dependency:
Needs biapy installed. Potentially as -e. We need to pass the /path/to/BiaPy_dir.

Call example: python -u /data/dfranco/datasets/synapses/scripts/calculate_detection_metrics_samia.py --input_pred_dir /data/dfranco/datasets/synapses/samia_results/new_run_256_patch_samia --input_gt_dir /data/dfranco/datasets/synapses/OCTO/test/raw.original --output_dir "/data/dfranco/datasets/synapses/samia_results/OUT/new_run_256_patch_samia" --BiaPy_dir /data/dfranco/BiaPy --tolerance 120
Author: Dani Franco Barranco
"""


import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import ast
from skimage.morphology import disk, dilation
import numpy as np
import h5py

parser = argparse.ArgumentParser(description="Convert semantic probabilities into points",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_pred_dir", "--input_pred_dir", required=True, help="Directory to the folder where the predicted points are stored")
parser.add_argument("-input_gt_dir", "--input_gt_dir", required=True, help="Directory to the folder where the GT points are stored")
parser.add_argument("-output_dir", "--output_dir", required=True, help="Output folder to store the final points")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
parser.add_argument("-tolerance", "--tolerance", default=120, type=int, help="Maximum distance far away from a GT point to consider a point as a true positive")
parser.add_argument("-resolution", "--resolution", default="(8,8,8)", type=str, help="Data resolution in (z,y,x)")
args = vars(parser.parse_args())

locations_path = "annotations.locations"
# resolution_path = 'volumes.raw'
partners_path = "annotations.presynaptic_site.partners"
id_path = "annotations.ids"

# Call example: python -u /data/dfranco/datasets/synapses/scripts/calculate_detection_metrics_samia.py --input_pred_dir /data/dfranco/datasets/synapses/samia_results/new_run_256_patch_samia --input_gt_dir /data/dfranco/datasets/synapses/OCTO/test/raw.original --output_dir "/data/dfranco/datasets/synapses/samia_results/OUT/new_run_256_patch_samia" --BiaPy_dir /data/dfranco/BiaPy --tolerance 120

sys.path.insert(0, args['BiaPy_dir'])
from biapy.engine.metrics import detection_metrics
from biapy.data.data_3D_manipulation import read_chunked_nested_data


# Check data resolution arg
resolution = args['resolution']
try:
    resolution = ast.literal_eval(resolution)
except:
    raise ValueError("'resolution' invalid: it must be a string like '(5,15,15)'")

print("Processing {} folder . . .".format(args['input_pred_dir']))
pred_ids = sorted(next(os.walk(args['input_pred_dir']))[2])
gt_ids = sorted(next(os.walk(args['input_gt_dir']))[2])

# Read the GT coordinates from the H5 file
gt_filename = os.path.join(args['input_gt_dir'], gt_ids[0])
file, ids = read_chunked_nested_data(gt_filename, id_path)
ids = list(np.array(ids))
_, partners = read_chunked_nested_data(gt_filename, partners_path)
partners = np.array(partners)
_, locations = read_chunked_nested_data(gt_filename, locations_path)
locations = np.array(locations)

gt_pre_points, gt_post_points = {}, {}
for i in range(len(partners)):
    pre_id, post_id = partners[i]
    pre_position = ids.index(pre_id)
    post_position = ids.index(post_id)
    pre_coord = locations[pre_position] // resolution
    post_coord = locations[post_position] // resolution
    if str(pre_coord) not in gt_pre_points:
        gt_pre_points[str(pre_coord)] = pre_coord
    if str(post_coord) not in gt_post_points:
        gt_post_points[str(post_coord)] = post_coord
gt_pre_points = list(gt_pre_points.values())
gt_post_points = list(gt_post_points.values())

if isinstance(file, h5py.File):
    file.close()

# Read the predicted pre coordinates from the CSV file
pred_pre_csv_path = os.path.join(args['input_pred_dir'], "pred_pre_locations.csv")

df_pred = pd.read_csv(pred_pre_csv_path)
zcoords = df_pred["Pre_Z"].tolist()
ycoords = df_pred["Pre_Y"].tolist()
xcoords = df_pred["Pre_X"].tolist()
# pred_pre_coordinates = [[z, y, x] for z, y, x in zip(zcoords, ycoords, xcoords)]
pred_pre_coordinates = [[max(int(z//8)-1,0), max(int(y//8)-1,0), max(int(x//8)-1,0)] for z, y, x in zip(zcoords, ycoords, xcoords)]

# Calculate detection metrics
if len(pred_pre_coordinates) > 0:
    d_metrics, gt_assoc, fp = detection_metrics(
        gt_pre_points,
        pred_pre_coordinates,
        true_classes=None,
        pred_classes=[],
        tolerance=args['tolerance'],
        resolution=resolution,
        bbox_to_consider=[],
        verbose=True,
    )
    print("Detection metrics (pre points): {}".format(d_metrics))

    # Save csv files with the associations between GT points and predicted ones
    os.makedirs(args['output_dir'], exist_ok=True)
    gt_assoc.to_csv(
        os.path.join(
            args['output_dir'],
            os.path.splitext(gt_ids[0])[0] + "_pre_gt_assoc.csv",
        )
    )
    fp.to_csv(
        os.path.join(
            args['output_dir'],
            os.path.splitext(gt_ids[0])[0] + "_pre_fp.csv",
        )
    )


# Read the predicted post coordinates from the CSV file
pred_pre_csv_path = os.path.join(args['input_pred_dir'], "pred_post_locations.csv")

df_pred = pd.read_csv(pred_pre_csv_path)
zcoords = df_pred["Post_Z"].tolist()
ycoords = df_pred["Post_Y"].tolist()
xcoords = df_pred["Post_X"].tolist()
# pred_post_coordinates = [[z, y, x] for z, y, x in zip(zcoords, ycoords, xcoords)]
pred_post_coordinates = [[max(int(z//8)-1,0), max(int(y//8)-1,0), max(int(x//8)-1,0)] for z, y, x in zip(zcoords, ycoords, xcoords)]

# Calculate detection metrics
if len(pred_post_coordinates) > 0:
    d_metrics, gt_assoc, fp = detection_metrics(
        gt_post_points,
        pred_post_coordinates,
        true_classes=None,
        pred_classes=[],
        tolerance=args['tolerance'],
        resolution=resolution,
        bbox_to_consider=[],
        verbose=True,
    )
    print("Detection metrics (post points): {}".format(d_metrics))

    # Save csv files with the associations between GT points and predicted ones
    os.makedirs(args['output_dir'], exist_ok=True)
    gt_assoc.to_csv(
        os.path.join(
            args['output_dir'],
            os.path.splitext(gt_ids[0])[0] + "_post_gt_assoc.csv",
        )
    )
    fp.to_csv(
        os.path.join(
            args['output_dir'],
            os.path.splitext(gt_ids[0])[0] + "_post_fp.csv",
        )
    )


print("FINISH!!")