Installation
============

Conda installation
------------------

- Install Anaconda by following steps in this `link <https://docs.anaconda.com/free/anaconda/install/index.html>`_
    -(Alternative: Install `Miniconda <https://docs.anaconda.com/free/miniconda/miniconda-install/>`_ )
*NB: Don't forget conda to your $PATH/bashrc/bash_profile*

Create the conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..  code-block:: console

    $ git clone https://github.com/Mohinta2892/catena.git
    $ cd catena/local_shape_descriptors/conda_env
    $ conda env create -f /path/to/the/environment.yml -n funkelsd

*NB: if -n `env_name` is not provided, the environment will be created as `funkelsd_test` as specified in the `environment.yml`.*

Verify package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can verify manually the existence of all packages as in the `environment.yml`. Run in `console`:

..  code-block:: console

    $ conda list


Docker installation
------------------
- Install Docker by following steps in this `link <https://docs.docker.com/engine/install/>`_

Downloading Docker images from DockerHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..  code-block:: console

    $ docker pull mohinta2892/lsd_sheridan_pytorch:22.01-py3
