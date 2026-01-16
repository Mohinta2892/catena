# Cookbook: Train a 3D MTLSD (LSD) model on CREMI (HDF → Zarr → Train)

This cookbook walks you end-to-end through:

1. Downloading one or more **CREMI `.hdf`** files
2. Converting them to the **exact Zarr layout** this repo expects (including `labels_mask`)
3. Putting them into the **expected folder structure**
4. Training a **3D multi-task LSD model (MTLSD)** with the provided pipeline

> **Why this is easier than “standard” setups:**
> The repo includes utilities to (a) create the on-disk structure, and (b) convert HDF5 → Zarr while automatically generating `volumes/labels/labels_mask`, which is **mandatory for training**. 

---

## What the code expects (don’t skip)

### 1) Data must be Zarr (currently)

Training/inference scripts **only accept `.zarr`** at the moment, and this has been tested on **zarr2** format.

### 2) Inside each `*.zarr`, the datasets must exist at these paths

Your Zarr must contain:

* `volumes/raw`
* `volumes/labels/neuron_ids`
* `volumes/labels/labels_mask` (1 = segmentation present, 0 = background / no seg)

This is exactly what the training pipeline reads via `ZarrSource(...)`.

---

## Step 0 — Get the code

```bash
git clone https://github.com/Mohinta2892/catena.git
cd catena
git checkout cookbooks
```

---

## Step 1 — Create an environment (Conda **or** Docker)

### Option A: Conda (recommended for most users)

The docs provide a simple conda install flow: clone → create env from `environment.yml`. 

```bash
cd catena/local_shape_descriptors/conda_env
conda env create -f environment.yml -n funkelsd
conda activate funkelsd
```

> **Version note:** for the data tools + training stack, the repo explicitly warns to use **python >=3.8 and <3.11** for now. 

### Option B: Docker / Apptainer (fastest “it just works” path)

A ready Docker workflow exists (including NVIDIA GPU support). 

Example pull:

```bash
docker pull mohinta2892/lsd_sheridan_pytorch:24.04-py3
```

Example run pattern (mount a “home” folder into the container):

```bash
docker run --shm-size 128gb --gpus all --pids-limit -1 -it \
  -u `id -u`:`id -g` -v `pwd`:`pwd` -w `pwd` \
  -v /path/to/your/home:/home --network=host \
  mohinta2892/lsd_sheridan_pytorch:24.04-py3
```

That `/home` mount is useful because several configs default to a `HOME`-rooted layout.

---

## Step 2 — Create the on-disk directory structure Catena expects

The repo includes a helper that creates this layout:

* `<base_dir>/<DOMAIN>/data_3d/train`
* `<base_dir>/<DOMAIN>/data_3d/test`
* plus 2D equivalents and a shared `preprocessed/` folder

Pick a base directory that matches your config. A common choice is:

* `HOME`: a “workspace root” (big disk)
* `DATA_DIR_PATH`: where you keep datasets under HOME
* `BRAIN_VOL`: dataset/domain name (here: `CREMI`)

For example, if you plan:

* `HOME=/data/lsd_home`
* `DATA_DIR_PATH=catena/data`

Then `base_dir=/data/lsd_home/catena/data`.

Run:

```bash
cd catena/local_shape_descriptors
python data_utils/download_data/create_dir_organisation.py /data/lsd_home/catena/data CREMI
```

This will create (among others):

```
/data/lsd_home/catena/data/
  CREMI/
    data_3d/
      train/
      test/
    data_2d/
      train/
      test/
  preprocessed/
```



---

## Step 3 — Download CREMI HDF5 files

CREMI typically comes as one or more `.hdf` files (e.g., sample volumes). For playing around, please download the data from here 2 [CREMI volumes.](https://huggingface.co/datasets/Mohinta2892/Catena_datasets/blob/main/cremi_3d_set_offset.tar.xz) 

Put your downloaded `.hdf` files into a staging directory, e.g.:

```bash
mkdir -p /data/lsd_home/cremi_hdf
# copy sample_A_*.hdf etc into /data/lsd_home/cremi_hdf
```

---

## Step 4 — Convert HDF5 → Zarr (and auto-generate `labels_mask`)

Use the provided conversion script:

```bash
cd catena/local_shape_descriptors/data_utils/download_data
python hdf_to_zarr.py -d /data/lsd_home/cremi_hdf -od /data/lsd_home/cremi_zarr_out
```

This is what it should show. Data in CREMI site do not have `offset` set. Check troubleshooting tips if you can't get it work.
```bash
Iterating over datasets hdf:: 100%|█| 9/9 [00:11<00:00,  1.24s/it,  
Iterating over datasets hdf:: 100%|█| 9/9 [00:10<00:00,  1.17s/it,  
```

What it does (important):

* Converts **all datasets** and attempts to keep attrs
* If `volumes/labels/neuron_ids` exists, it generates:

  * `volumes/labels/labels_mask` = 0 where neuron_ids==0, else 1
  * copies `offset` and `resolution` attrs onto the mask


> **Heads up:** the repo warns this simple converter is **not suited for large datasets**; for big volumes, a daisy-driven approach is recommended. 

### Quick sanity check: does the Zarr contain the required keys?

(Any quick Zarr inspector works; the important bit is: do you see the 3 paths?)

Required:

* `volumes/raw`
* `volumes/labels/neuron_ids`
* `volumes/labels/labels_mask`

---

## Step 5 — Place the converted `.zarr` files into the training folder

Move (or symlink) the converted `.zarr` outputs into:

```
<base_dir>/CREMI/data_3d/train/
```

Example:

```bash
mv /data/lsd_home/cremi_zarr_out/*.zarr /data/lsd_home/catena/data/CREMI/data_3d/train/
```

At this point, training will discover samples by globbing `*.zarr` inside your train folder (this pattern is used throughout the training codepaths). 

---

## Step 6 — Configure CREMI training (minimal edits)

The repo provides a dataset-specific config for CREMI (`config_cremi.py`) and sets CREMI as an anisotropic dataset with voxel size `(40, 4, 4)` and CREMI-tuned shapes. 

**What you almost always must change:**

* `_C.DATA.HOME`
* `_C.DATA.DATA_DIR_PATH`

In `local_shape_descriptors/config/config_cremi.py`, set these to match your machine/disk layout. 

Also confirm:

* `_C.DATA.BRAIN_VOL = "CREMI"`

Ensure these are set like this:
* `_C.DATA.FIB = 0`  # Means FIBSEM isotropic data; **CREMI is NOT**
* `_C.DATA.WITH_MITO = 0` # 0: NO MITO, 1: MITO ONLY, 2: MITO + LSD + AFF (MTLSDMITO)

---

## Step 7 — Train 3D MTLSD (LSD + Affinities multitask)

The training pipeline reads:

* `volumes/raw`
* `volumes/labels/neuron_ids`
* `volumes/labels/labels_mask`

and will run MTLSD when configured accordingly (MTLSD is a first-class model type in the codebase).

From `catena/local_shape_descriptors`:

```bash
python trainer.py -c config/config_cremi_test.py
```

**What gets created automatically:**

* checkpoint folder
* log folder
* snapshot folder
  (these are derived from your config and created at runtime)

CREMI config builds output paths under `HOME` like:

* `.../lsd_checkpoints/...`
* `.../lsd_logs/...`
* `.../lsd_snapshots/...` 

Also note: training/prediction log files under `local_shape_descriptors/logs/` can get large over long runs. 

---

## Troubleshooting (the ones that bite most often)

### CREMI HDF to ZARR

The hdf to zarr conversion requires `offset` set in CREMI hdf files per dataset.
However, the data you download from the CREMI website does not have `offset` set as attributes.
You can set them by hand in the hdf like below:

```bash
import h5py
f = h5py.File("/path/to/file.hdf")
f["volumes/labels/neuron_ids"].attrs["offset"] = (0,0,0) # (0,0,0) is only valid for the cropped datasets
f["volumes/raw"].attrs["offset"] = (0,0,0)
f["volumes/labels/labels_mask"].attrs["offset"] = (0,0,0)
```

### “No files detected” / empty training set

This is almost always data layout. Confirm:

* your `.zarr` files are physically inside
  `.../CREMI/data_3d/train/`
* the Zarr contains the required dataset keys (`volumes/raw`, `volumes/labels/neuron_ids`, `volumes/labels/labels_mask`)

### “It crashes when sampling batches”

Common cause: `labels_mask` is missing or all zeros in the region being sampled. This repo uses `labels_mask` to bias sampling toward valid GT regions.

### Python version weirdness

If you’re on python 3.11+, you may hit “unforeseen problems” according to the repo notes—use 3.8–3.10.12 for now. 

---

## (Optional) Next steps after training

Once you have a checkpoint, you can run affinity/LSD prediction using the provided predictors (single-process or daisy blockwise) and then proceed to instance segmentation. The inference docs include the expected layout for checkpoints and test volumes.

