# Use the NVIDIA PyTorch base image that comes with CUDA; you can change this to anything you want!
FROM nvcr.io/nvidia/pytorch:23.09-py3

# Set the working directory
WORKDIR /env_install

# Copy the lsd requirements
COPY requirements.txt .

# Install requirements (no need to install torch or cuda, the base docker comes with it)
RUN pip install -r requirements.txt

# Install miniconda (Credit: https://fabiorosado.dev/blog/install-conda-in-docker/)
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Plant-Seg for Instance Segmentation in a conda env with mamba
RUN conda install -c conda-forge mamba

# GPU expected - hopefully this won't interfere with existing pytorch
RUN mamba create -y -n plant-seg -c pytorch -c nvidia -c conda-forge -c lcerrone plantseg pytorch-cuda=12.1

# If not available, comment the above line and run this line instead for a CPU-only install
# RUN mamba create -n plant-seg -c pytorch -c conda-forge -c lcerrone plantseg cpuonly

## Install watershed implementation of Aleksandar Zlateski and Chandan Singh
## Make a different env for this since we don't want dependency leaks due to old packages
RUN apt-get update && apt-get install -y libboost-dev

# Use python=3.9 as ERROR:Fails to build on Python 3.11 - longintrepr.h: No such file or directory
# Git Issue: https://github.com/aio-libs/aiohttp/issues/6600
RUN conda create -n waterz python=3.9

# This correctly activates the waterz env
RUN echo "source activate waterz" > ~/.bashrc

# Adds waterz env to the PATH variable
ENV PATH /opt/conda/envs/waterz/bin:$PATH

# Install from the git
RUN python -m pip install git+https://github.com/funkey/waterz.git

# Change the current directory to home
WORKDIR /home

# Optionally, you can add additional setup or run commands here
# CMD ["python", "your_script.py"]

## How we run the docker in terminal after building
#nvidia-docker run -it --network=host -e WANDB_API_KEY="local-4bafb5769b38a01c12c8c842a9166ea27ec164d4" -e WANDB_BASE_URL=http://localhost:8080 -v /media/samia/DATA/PhD/codebases/restructured_packages:/home -v /media/samia/DATA/ark/connexion:/mnt docker.io/library/lsd_pytorch2:v1

