### Set up Anaconda/Miniconda
- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)/[Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) as per your OS.

  
**NB: Don't forget to add conda to your $PATH/bashrc/bash_profile**

>[!WARNING]
> Our conda_env YAML contains both `conda` and `pip` dependencies.
> We have tested building conda envs with this file on Linux systems running Ubuntu>=20.04.

### Build the conda env for Local Shape Descriptors
```shell
conda env create -f /catena/local_shape_descriptors/conda_env/environment.yml -n funkelsd
```
>[!NOTE]
> You can choose any name for the environment with `-n`. Default if none specified is `funkelsd_test`.

<details><summary>Remove the conda environment</summary>
<br>

```shell
conda remove -n funkelsd --all
```
</details>
