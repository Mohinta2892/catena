### Set up Anaconda/Miniconda
- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)/[Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) as per your OS.

  
**NB: Don't forget to add conda to your $PATH OR bashrc OR bash_profile**

>[!WARNING]
> Our conda_env YAML contains both `conda` and `pip` dependencies.
> We have tested building conda envs with this file on Linux systems running Ubuntu>=20.04.

>[!IMPORTANT]
> Please install `libboost-dev` in your Ubuntu/Linux OS environment.
> It's a mandatory requirement for `funlib.segment`. <br>
```bash

    sudo apt-get update
    sudo apt-get install libboost-dev
```

### Build the conda env for Local Shape Descriptors
- Choose [complete_py310_environment.yml](https://github.com/Mohinta2892/catena/blob/dev/local_shape_descriptors/conda_env/complete_py310_environment.yml) for installing python=3.10 in machines with latest CUDA drivers

```shell
conda env create -f /catena/local_shape_descriptors/conda_env/<environment.yml> -n funkelsd
```
`waterz` is currently not part part of the `complete_py310_environment.yml`. You can try running `pip install git+https://github.com/funkey/waterz.git` **after the conda env** is built.
If it does not work please following the instructions in the [Troubleshooting below](#Troubleshoting).

>[!NOTE]
> You can choose any name for the environment with `-n`. Default if none specified is `funkelsd_test`.

<details><summary>Remove the conda environment</summary>
<br>

```shell
conda remove -n funkelsd --all
```
</details>

### Troubleshooting

- We have an ongoing issue with install waterz in newer machines. First remove it from the `.yml` env file and then please follow the instructions [here](https://github.com/Mohinta2892/catena/tree/dev/local_shape_descriptors/install_src).
  
