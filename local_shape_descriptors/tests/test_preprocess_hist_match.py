import sys
import os

# add current directory to path and allow absolute imports
sys.path.insert(0, '../')
from data_utils.preprocess_volumes.histogram_match import match_histograms
from config.config import get_cfg_defaults

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    # can be used to override pre-defined settings
    # TODO test with explicit path for example setups: ssTEM CREMI, FIBSEM: Hemibrain
    if os.path.exists("./experiment.yaml"):
        cfg.merge_from_file("experiment.yaml")
    
    data_path = "/media/samia/DATA/ark/connexion/data"
    # dataset names inside the data_path (all combinations will be computed)
    datasets = ['HEMI', "TREMONT"]
    dimensionality = ["data_3d"]
    cfg.PREPROCESS.DATASETS_TO_COPY = ["volumes/labels/neuron_ids", "volumes/labels/labels_mask"]
    match_histograms(data_path=data_path, datasets=datasets, dimensionality=dimensionality, cfg=cfg)

    # 2D
    # dimensionality = ["data_2d"]
    # cfg.PREPROCESS.DATASETS_TO_COPY = ["volumes/labels", "volumes/labels_mask"]
    # match_histograms(data_path=data_path, datasets=datasets, dimensionality=dimensionality, cfg=cfg)
