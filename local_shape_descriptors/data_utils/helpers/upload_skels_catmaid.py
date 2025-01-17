"""
Upload skeletons to catmaid.
This script currently requires a .pkl file which contains TreeNeuron(skels) for selected segmentations that we wish to push into a CATMAID project.
It will be later updated to a generic script.

Author: Samia Mohinta
Affiliation: Cambridge University, UK
"""

import pymaid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import zarr
import dask.array
from datetime import datetime
from tqdm import tqdm
import sys
from joblib import Parallel, delayed
import json
import pickle

url = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/"
token = "90f00590555eb256f256d1062e38521cbe180293"
name = "smohinta"
password = "headset-recovery-handshake"
project_id = 19  # MR1.4-3 clone

# Always to a specific project

rm = pymaid.CatmaidInstance(url, token, name, password, project_id, caching=False)
# rm = pymaid.CatmaidInstance(url, token, name, password, caching=False)

with open("/media/samia/DATA/mounts/cephfs/catena/mr143_skels/KC_skeletons.pkl", 'rb') as f:
    skeletons_of_interest = pickle.load(f)

n = skeletons_of_interest[2:]
resp_all = dict()

for x in n:
    resp = pymaid.upload_neuron(x, source_type=None, remote_instance=rm)  # passing remote instance is mandatory for us.
    resp_all.update(resp)
    print(resp)

with open(
        '/media/samia/DATA/ark/dan-samia/lsd/code-dev-samia/data-download/winding_synapse_extraction/pymaid_import_resp_kcs.json',
        'w') as f:
    json.dumps(resp_all, f)
