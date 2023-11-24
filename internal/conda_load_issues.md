# Issues with loading conda in Cardona-GPU1/CPU1:

/path/to/conda: Device Busy or Bad interpreter

## Solution:
- Login to gpu1 and run `echo $SHELL`. Output should be `/bin/tcsh`.
- Run ` /opt/miniconda3/bin/conda env list`

Do you see something as below:
```
# conda environments:
#
                         /lmb/home/smohinta/anaconda3
                         /lmb/home/smohinta/anaconda3/envs/lsd
                         /lmb/home/smohinta/anaconda3/envs/n5mrc
                         /lmb/home/smohinta/anaconda3/envs/paintera
                         /lmb/home/smohinta/anaconda3/envs/synful
base                  *  /opt/miniconda3
```
This means that your environments have become headless. So you cannot run `conda activate lsd` anymore. 
We need to give them back their env names.

- RUN `/opt/miniconda3/bin/conda config --append envs_dirs /lmb/home/smohinta/anaconda3/envs`
- RUN again ` /opt/miniconda3/bin/conda env list`

Output should look as below, with names reinstated:
```
# conda environments:
#
                         /lmb/home/smohinta/anaconda3
lsd                      /lmb/home/smohinta/anaconda3/envs/lsd
n5mrc                    /lmb/home/smohinta/anaconda3/envs/n5mrc
paintera                 /lmb/home/smohinta/anaconda3/envs/paintera
synful                   /lmb/home/smohinta/anaconda3/envs/synful
base                  *  /opt/miniconda3
```

- RUN `/opt/miniconda3/bin/conda init tcsh`
- RUN `/opt/miniconda3/bin/conda init bash`
- Exit from gpu1 and relogin.

## Test
- RUN `conda`
Output should now look like:

```
conda
usage: conda [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.

Options:

positional arguments:
  command
    clean        Remove unused packages and caches.
    config       Modify configuration values in .condarc. This is modeled
                 after the git config command. Writes to the user .condarc
                 file (/lmb/home/smohinta/.condarc) by default.
    create       Create a new conda environment from a list of specified
                 packages.
    help         Displays a list of available conda commands and their help
                 strings.
    info         Display information about current conda install.
    init         Initialize conda for shell interaction. [Experimental]
    install      Installs a list of packages into a specified conda
                 environment.
    list         List linked packages in a conda environment.
    package      Low-level conda package utility. (EXPERIMENTAL)
    remove       Remove a list of packages from a specified conda environment.
    uninstall    Alias for conda remove.
    run          Run an executable in a conda environment. [Experimental]
    search       Search for packages and display associated information. The
                 input is a MatchSpec, a query language for conda packages.
                 See examples below.
    update       Updates conda packages to the latest compatible version.
    upgrade      Alias for conda update.

optional arguments:
  -h, --help     Show this help message and exit.
  -V, --version  Show the conda version number and exit.

conda commands available from other packages:
  env
```
- RUN `conda env list`
This should show you all your environment names.

- RUN `conda activate {env_name}`
