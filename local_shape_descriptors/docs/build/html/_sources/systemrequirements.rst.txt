System Requirements
===================

**Operating System**: Ubuntu 20.4 or above
   - Code has been tested on Ubuntu 22.04.3 LTS and Ubuntu 20.04.6 LTS (dockerised)

**Python Version**: 3.8 or above
   - Code has been developed and tested on Python 3.8, 3.9, and 3.10 (note: using Python 3.11 may result in some issues)

**Anaconda/Miniconda Version**: 23.1.0 or above
   - Code has been developed and tested on Conda version 23.1.0

**RAM**: 64GB or above
   - Code has been developed and tested on machines with:
   	- 128GB of RAM (Supermicro) &
   	- 504GB of RAM (Nvidia-DGX)

**CPU**: Any modern AMD/Intel Processors
   - Code has been developed and tested on two different machines with:
       - `AMD EPYC 7513 32-Core Processor` (Supermicro) &
       - `Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz` (Nvidia-DGX)

**GPU**: Latest cards with minimum 12GB GPU memory
   - Code has been developed and tested on machines with: `RTX 3090 24GB` (Supermicro) and `Nvidia Titan XP 12GB` x 8 cards (Nvidia-DGX)
   
   
Databases
----------
**MongoDB** :
For multi-worker inference, please install `MongoDB <https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/#std-label-install-mdb-community-ubuntu>`_. Our code has been tested with MongoDB `v6.0.5`.

**Postgres**:
We are currently extending support to `PostGreSQL <https://www.postgresql.org/download/linux/ubuntu/>`_. Our preliminary tests have been conducted on PostgreSQL `v14.9`.


	
Containerisation of application environments
---------------------------------------------

| Our development and test environments have been containerised using dockers and apptainers. 
| We highly recommend installing `Docker <https://www.docker.com/get-started>`_ : v24.0.6 or above.	

| `Apptainers (formerly Singularity) <https://apptainer.org/docs/admin/main/installation.html#>`_ can be your choice of containerisation too. 
| Our apptainers have docker images as bases, please build them using the provided `.def` files.
| We have tested the builds and runs on versions 1.2.3.
