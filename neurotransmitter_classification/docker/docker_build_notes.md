#### Building the docker in local 

>[!WARNING]
>Change the path to the conda_env file before building the docker


```bash
cd neurotransmitter_classification/docker
docker build -t synister . --no-cache
```
`--no-cache`: is optional should be used when building from scratch

There are a few packages commented out in the `env.yml` file.
Some are unnecessary, hence we have done so.

However, we need to load `funlib.segment` and `funlib.evaluate`.
`funlib.segment` requires installing `libboost-all-dev`.

Please run the below commands to run after the docker is built:

- Run the docker:
```bash
docker run -it synister:latest 
```

- Load conda env in the docker:
```bash
conda activate synister
```

- Install `libboost`
```bash
conda install -c conda-forge boost
```

- Install `funlib.segment` and `funlib.evaluate`
```bash
pip install git+https://github.com/funkelab/funlib.evaluate.git
pip install git+https://github.com/funkelab/funlib.segment.git
```


