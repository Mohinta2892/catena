#!/bin/bash

### Run chmod +x install_waterz.sh
### ./install_waterz.sh your_env_name path/to/waterz.zip path/to/waterz-dist-info.zip
### Remove the `.zip` from the above paths, should be like  path/to/waterz
# Download waterz zips if you have not git cloned the catena
# wget https://github.com/Mohinta2892/catena/tree/dev/local_shape_descriptors/install_src/waterz.zip
# wget https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/install_src/waterz-0.9.5.dist-info.zip


# Check if correct number of arguments is passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 ENV_NAME WATERZ_ZIP WATERZ_DIST_ZIP"
    exit 1
fi

# Assign command-line arguments to variables
ENV_NAME=$1
WATERZ_ZIP=$2
WATERZ_DIST_ZIP=$3

# Activate the conda environment
conda init bash
conda activate $ENV_NAME

# Check the versions of numpy and cython
NUMPY_VERSION=$(python -m pip show numpy | grep Version | awk '{print $2}')
CYTHON_VERSION=$(python -m pip show cython | grep Version | awk '{print $2}')

if [ "$NUMPY_VERSION" != "1.24.4" ]; then
    echo "Error: numpy version is $NUMPY_VERSION, expected 1.24.4"
    exit 1
fi

if [ "$CYTHON_VERSION" != "0.29.34" ]; then
    echo "Error: cython version is $CYTHON_VERSION, expected 0.29.34"
    exit 1
fi

# Unzip the files
unzip $WATERZ_ZIP #-d $(dirname $WATERZ_ZIP)
unzip $WATERZ_DIST_ZIP #-d $(dirname $WATERZ_DIST_ZIP)

# Find the path to site-packages
SITE_PACKAGES_PATH=$(python -m pip show wandb | grep Location | awk '{print $2}')

# Sanity check that the path exists
if [ ! -d "$SITE_PACKAGES_PATH" ]; then
  echo "Site-packages path does not exist: $SITE_PACKAGES_PATH"
  exit 1
fi

# Copy waterz and waterz-dist-info into the site-packages
cp -r $(dirname $WATERZ_ZIP)/waterz $(dirname $WATERZ_DIST_ZIP)/waterz-0.9.5.dist-info $SITE_PACKAGES_PATH

# Sanity check that pip has access to waterz in the new environment
source activate $ENV_NAME
python -m pip show waterz

echo "Script execution completed."


