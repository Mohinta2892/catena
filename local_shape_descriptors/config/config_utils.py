def update_cfg_from_args(cfg, args):
    """UPDATE YACS with ARGS passed arguments. Default both remains the same"""
    # Update SYSTEM section
    cfg.SYSTEM.NUM_GPUS = args.num_gpus
    cfg.SYSTEM.NUM_WORKERS = args.num_workers
    cfg.SYSTEM.CACHE_SIZE = args.cache_size
    cfg.SYSTEM.VERBOSE = args.verbose

    # Update DATA section
    cfg.DATA.HOME = args.home
    cfg.DATA.DATA_DIR_PATH = args.data_dir_path
    cfg.DATA.BRAIN_VOL = args.brain_vol
    cfg.DATA.TRAIN_TEST_SPLIT = args.train_test_split
    cfg.DATA.FIB = args.fib
    cfg.DATA.DIM_2D = args.dim_2d
    cfg.DATA.OUTFILE = args.outfile
    cfg.DATA.INVERT_PRED_AFFS = args.invert_pred_affs

    # Update PREPROCESS section (if DIM_2D is True)
    if args.dim_2d:
        cfg.PREPROCESS = CN()
        cfg.PREPROCESS.EXPORT_2D_FROM_3D = args.export_2d_from_3d
        cfg.PREPROCESS.HISTOGRAM_MATCH = args.histogram_match
        cfg.PREPROCESS.USE_WANDB = args.use_wandb

    # Update TRAIN section
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NEIGHBORHOOD = args.neighborhood
    cfg.TRAIN.NEIGHBORHOOD_2D = args.neighborhood_2d
    cfg.TRAIN.LR_NEIGHBORHOOD = args.lr_neighborhood
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.SAVE_EVERY = args.save_every
    cfg.TRAIN.DEVICE = args.device
    cfg.TRAIN.INITIAL_LR = args.initial_lr
    cfg.TRAIN.LR_BETAS = tuple(args.lr_betas)
    cfg.TRAIN.MODEL_TYPE = args.model_type
    cfg.TRAIN.CHECKPOINT = args.checkpoint

    # Update Model-specific configurations (Add your model-specific updates here)
    return cfg
