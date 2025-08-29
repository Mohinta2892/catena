Running inference with Pretrained networks
==========

Data Organisation
---------

Since you are running only inference, you need to make sure to have your data and checkpoints organised (in **.zarr** format) as below. If you had run training, all necessary folders are
created by the code.


::

  home
    - catena
        - data
            - HEMI
                - data_3d
                    - train
                    - test  # your files must go under this folder before you run inference
                - data_2d
                    - train
                    - test  # use this if you want to run inference with 2D model

    - lsd_outputs # outputs will be saved here
    - lsd_checkpoints 
        - MTLSD_3D
            - run_1
                - model_checkpoint_300000

Download Checkpoints
---------
We share our best MTLSD model's checkpoint `here <https://www.dropbox.com/scl/fo/7i0fubgdredt1z9jguz8x/APceQkRdquwTEXEVgL4vJWo?rlkey=ml5aqoha24vxdlu09obo07dyl&st=4f1lmiii&dl=0>`_.

Edit the config file `config_predict.py <https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/config/config_predict.py>`_
---------

You must modify the following settings:

.. code-block:: python

    _C.DATA.HOME = "path/to/home"  # options: /home; /local/path/till/catena
    _C.DATA.DATA_DIR_PATH = "catena/data"  # where the code resides and data should too
    _C.DATA.BRAIN_VOL = "HEMI"

If you are using Mongo and have it installed, change the following:

.. code-block:: python

    _C.DATA.DB_HOST = ''  # point to your DB host, on local machines this is ``localhost:27017`` or ``127.0.0.1:27017``
    _C.DATA.DROP_DS_MONGOTABLE = False  # preserves both the dataset and the Mongo Collection

Change the GPU device and Model type:

.. code-block:: python

    _C.TRAIN.DEVICE = "cuda:0"  # "cuda" if torch.cuda.is_available() else "cpu"
    _C.TRAIN.MODEL_TYPE = "MTLSD"  # options: `MTLSD`, `ACLSD`, `ACRLSD`, `LSD`, `AFF`

Point to the checkpoint:

.. code-block:: python

    _C.TRAIN.CHECKPOINT = f"{_C.DATA.HOME}/lsd_checkpoints/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/run-aclsd-together/model_checkpoint_300000" # path to your checkpoint as you have organised above

**Adjust shapes of input and voxel size for Isotropic Data or Running with Model Pretrained on Isotropic Data:**

.. code-block:: python

    _C.MODEL_ISO.INPUT_SHAPE = (196, 196, 196)  # default in (ZYX)
    _C.MODEL_ISO.INPUT_SHAPE_2D = (196, 196)  # default
    _C.MODEL_ISO.OUTPUT_SHAPE = (72, 72, 72)  # default based on (196)^3 input shape, you will be shown what this should be when you run inference
    _C.MODEL_ISO.OUTPUT_SHAPE_2D = (72, 72)  # default based on (196)^2 input shape, you will be shown what this should be when you run inference
    _C.MODEL_ISO.VOXEL_SIZE = (8, 8, 8)  # default for iso models
    _C.MODEL_ISO.VOXEL_SIZE_2D = (12, 12)  # default for iso models
    _C.MODEL_ISO.GROW_INPUT = (36, 36, 36)  # set this to (0,0,0) if you do not want to use context, can cause problems with small volumes
    _C.MODEL_ISO.GROW_INPUT_2D = (36, 36)  # set this to (0,0) if you do not want to use context, can cause problems with small volumes

**Adjust shapes of input and voxel size for Anisotropic Data or Running with Model Pretrained on Anisotropic Data:**

.. code-block:: python

    _C.MODEL_ANISO.INPUT_SHAPE = (132, 268, 268)  # default in (ZYX)
    _C.MODEL_ANISO.INPUT_SHAPE_2D = (268, 268)  # default
    _C.MODEL_ANISO.OUTPUT_SHAPE = (72, 144, 144)  # default based on (132, 268, 268) input shape, you will be shown what this should be when you run inference
    _C.MODEL_ANISO.OUTPUT_SHAPE_2D = (144, 144)  # default based on (268, 268) input shape, you will be shown what this should be when you run inference
    _C.MODEL_ANISO.VOXEL_SIZE = (40, 4, 4)  # default for aniso models
    _C.MODEL_ANISO.VOXEL_SIZE_2D = (4, 4)  # default for aniso models
    _C.MODEL_ANISO.GROW_INPUT = (36, 36, 36)  # set this to (0,0,0) if you do not want to use context, can cause problems with small volumes
    _C.MODEL_ANISO.GROW_INPUT_2D = (36, 36)  # set this to (0,0) if you do not want to use context, can cause problems with small volumes

Use `predicter.py <https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/predicter.py>`_ for Affinity Prediction on Small Volumes (must fit in RAM)
---------

.. code-block:: python

  python predicter.py -c config_predict.py

.. note::
   :class: red-note

    This should display that names of the .zarr test input files. If no files are detected, there is a problem with data organisation. 
    You have a predict log saved under ``local_shape_descriptors/logs``.


Use `super_predicter_daisy.py <https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/predicter.py>`_  for Affinity Prediction on Large Volumes (runs chunk by chunk)
---------
.. code-block:: python

  python super_predicter_daisy.py -c config_predict.py

.. note::
   :class: red-note

    This should display that names of the .zarr test input files. If no files are detected, there is a problem with data organisation. 
    You have a predict log saved under ``local_shape_descriptors/daisy_logs/{zarr_file_name}``.

Instance Segmentation
------------

For Small volumes Watershed and Agglomeration happen at the same time
~~~~~~~~~~~~

.. code-block:: python

  python instance_segmenter.py -c config_predict.py

For Large volumes Watershed and Agglomeration have to be run separately
~~~~~~~~~~~~

.. note::
   :class: red-note

    MongoDB installation is mandatory for this to work.

Watershed:

.. code-block:: python

  python 02_extract_fragments_blockwise.py -c config_predict.py

Agglomerate:

.. code-block:: python

  python 03_agglomerate_blockwise.py -c config_predict.py

Extract Instances:

.. code-block:: python

   python 04_find_segments_full.py daisy_logs/config_0.yml
