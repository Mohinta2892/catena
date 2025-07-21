FROM continuumio/miniconda3

# Install essential build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    libboost-all-dev

# RUN apt-get install libboost-all-dev

# Copy the environment.yml file
COPY synister_environment_workstation.yml /tmp/environment.yml

# Create and activate the conda environment
RUN conda env create -f /tmp/environment.yml -n synister
RUN conda clean --all
#RUN conda init
#RUN conda activate synister
#RUN conda install -c conda-forge boost

# Activate your conda environment by default
SHELL ["conda", "run", "-n", "synister", "/bin/bash", "-c"]

# Set the default command
CMD ["/bin/bash"]
