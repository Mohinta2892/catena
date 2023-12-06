from __future__ import annotations
import os.path
import sys

from local_shape_descriptors.config.config import get_cfg_defaults

if __name__ == "__main__":
    # check if current dir is in sys path
    print(sys.path)
    cfg = config.get_cfg_defaults()
    if os.path.exists("./experiment.yaml"):
        cfg.merge_from_file("experiment.yaml")
    cfg.freeze()
    print(cfg)

