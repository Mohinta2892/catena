# Changes to code that have been made to run inference with pretrained TensorFlow networks

- Removed dependency on creating `experiment` folders to run the code. 
  Change in `predict_blockwise.py`, method `predict_blockwise(...)` lines 102-105.
- Modified `check_block` to find block status (line 331-332) to work with latest MongoDB and PyMongo in `predict_blockwise.py`.
- Replaced `daisy.call()` with `os.system(command)` in `predict_blockwise.py`.
- Added `ZarrWrite(raw)` to `p_setup52`'s `predict.py` in the output zarr. Can be added to other files if needed.
- Corrected `insert` command for `block_done_callback` in `predict.py`.
