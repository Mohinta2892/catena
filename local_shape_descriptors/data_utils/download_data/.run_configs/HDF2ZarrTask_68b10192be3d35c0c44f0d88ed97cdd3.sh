#!/bin/bash
#SBATCH -t 11:40:00
#SBATCH -p short
#SBATCH -c 1
#SBATCH --mem=2GB
#SBATCH -o .logs/HDF2ZarrTask_68b10192be3d35c0c44f0d88ed97cdd3_%j.out
#SBATCH -e .logs/HDF2ZarrTask_68b10192be3d35c0c44f0d88ed97cdd3_%j.err

python /media/samia/DATA/PhD/codebases/restructured_packages/local_shape_descriptors/data_utils/download_data/hdf_to_zarr_daisy.py run_worker .run_configs/HDF2ZarrTask_68b10192be3d35c0c44f0d88ed97cdd3.config