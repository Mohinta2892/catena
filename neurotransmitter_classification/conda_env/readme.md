# System Requirements

- Local Workstation Requirements on which the code has been developed and tested are listed [here](./local_workstation.md)
- Slurm Cluster Requirements where code has been tested for long jobs are listed  [here](./slurm_cluster.md)

# Conda ENV Installation 

- Please follow the anaconda/miniconda installation per your OS [here](https://www.anaconda.com/docs/getting-started/anaconda/install)
- You can then run either the `workstation yaml` (if running on a desktop) or the `slurm yaml` (if running on a HPC cluster) env files as below:
  ```bash
      conda env create -n <env_name> -f <path/to/env.yml>
  ```
