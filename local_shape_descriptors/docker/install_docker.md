# How to use LSD Docker & Apptainers (formerly Singularity)?

### Docker Installation and Image Pull/Build
1. Install [Docker Engine](https://docs.docker.com/engine/install/) and verify installation in your OS environment.
> [!IMPORTANT]
> Must [install and configure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) `nvidia-docker` to use `nvidia/cuda` GPUs for model training. 
2. Pulling/building docker images:
   1. You can pull the latest docker image from DockerHub. This will load the image onto your local memory.

      ```
      docker pull mohinta2892/lsd_sheridan_pytorch:22.01-py3
      ``` 

   2. You can build the docker image yourself:
        ``` 
         cd catena/local_shape_descriptors/docker
         docker build -t lsd_sheridan_pytorch:latest .
        ```
   `-t` tag the docker image with a `name` (lsd_sheridan_pytorch) and `version` (latest).
> [!TIP]
> You can pass Dockerfiles to docker build as ` docker build -f /path/to/Dockerfile.updated /catena/local_shape_descriptors `
> `/catena/local_shape_descriptors` is the context given to the docker. Meaning of [commands](https://docs.docker.com/reference/cli/docker/image/build/)

### Running the code within the docker

```
nvidia-docker run --shm-size 128gb --pids-limit -1 -it -u `id -u`:`id -g` -v `pwd`:`pwd` -w `pwd` -v /path/to/codendata:/home --network=host lsd_sheridan_pytorch:22.01-py3
```
<details>
<summary><strong>Command Explanation</strong></summary>

`nvidia-docker run` starts a new container using an NVIDIA Docker image. The additional options specify the container's resources, permissions, and environment:

- `--shm-size 128gb` sets the shared memory size available to the container to 128 gigabytes. This is useful for processes within the container that use shared memory for communication, like some data processing and machine learning tasks.

- `--pids-limit -1` removes any limitation on the number of process IDs (PIDs) that can be created within the container. This is useful for applications that spawn many processes.

- `-it` runs the container in interactive mode with a terminal attached, allowing you to interact with the command line inside the container.

- `-u \`id -u\`:\`id -g\`` sets the user ID and group ID inside the container to match the current user's UID and GID, promoting security by not running processes as the root user inside the container.

- `-v \`pwd\`:\`pwd\`` mounts the current working directory (`pwd` returns the present working directory's path) on the host machine to the same path inside the container, ensuring files and data are synchronised between the host and the container.

- `-w \`pwd\`` sets the working directory inside the container to match the host's current working directory, meaning when you start the container, it will execute commands from this directory.

- `-v /path/to/codendata:/home` mounts the host directory `/path/to/codendata` to `/home` inside the container, allowing the container to access and store data in this directory.

- `--network=host` configures the container to use the host's network stack, meaning the container shares the host's IP address and port namespace. This is useful for network-intensive applications or when the container needs to listen on the host's network interfaces directly.

- `lsd_sheridan_pytorch:22.01-py3` specifies the Docker image to use for the container. In this case, it's using the `latest` version of an image named `lsd_sheridan_pytorch:22.01-py3`.
</details>

### Apptainers Installation, Build and Run (*Optional*)
<details close> 
<summary><strong>Installation/Build</strong></summary>
<br>
1. Install <a href="https://apptainer.org/docs/admin/1.0/installation.html#">Apptainer</a> and verify in your OS environment.<br>
2. Build the apptainer:<br>
<code>sudo apptainer build lsd_sheridan_pytorch_2201py3.sif docker://mohinta2892/lsd_sheridan_pytorch:22.01-py3</code><br>
Customize your build (e.g., writable sandbox) by following instructions <a href="https://apptainer.org/docs/user/1.0/build_a_container.html">here</a>.<br>
<blockquote style="color: red;">
  <strong>WARNING:</strong> Apptainer has not been extensively tested yet across High-Performance-Computing environments. Hence, there may be issues.
</blockquote>
</details>

<details close> 
<summary><strong>Run apptainer</strong></summary>
<br>

```
apptainer run  --nv --bind /path/to/codendata:/home apptainer_image/lsd_sheridan_pytorch_2201py3.sif
```
</details>
