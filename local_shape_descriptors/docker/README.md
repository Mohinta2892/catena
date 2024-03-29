# How to use LSD Docker & Apptainers (formerly Singularity)?

### Docker Installation and Image Pull/Build
1. Install [Docker Engine](https://docs.docker.com/engine/install/) and verify installation in your OS environment.
2. Pulling/building docker images:
   1. You can pull the latest docker image from DockerHub:
      ```
      docker pull mohinta2892/lsd_sheridan_pytorch:22.01-py3
      ```
     This will load the image onto your local memory.
   2. You can build the docker image yourself:
     ``` 
      cd catena/local_shape_descriptors/docker
      docker build -t lsd_sheridan_pytorch:latest .
     ```
   -t tag the docker image with a name (lsd_sheridan_pytorch) and version (latest).
> [!TIP]
> You can pass Dockerfiles to docker build as ` docker build -f /path/to/Dockerfile.updated /catena/local_shape_descriptors `
> `/catena/local_shape_descriptors` is the context given to the docker. Meaning of [commands](https://docs.docker.com/reference/cli/docker/image/build/)

<details close> 
<summary><strong>Apptainer Installation and Image Pull/Build</strong></summary>
<br>
1. Install <a href="https://apptainer.org/docs/admin/1.0/installation.html#">Apptainer</a> and verify in your OS environment.<br>
2. Build the apptainer:<br>
<code>sudo apptainer build lsd_sheridan_pytorch_2201py3.sif docker://mohinta2892/lsd_sheridan_pytorch:22.01-py3</code><br>
Customize your build (e.g., writable sandbox) by following instructions <a href="https://apptainer.org/docs/user/1.0/build_a_container.html">here</a>.<br>
>[!WARNING]<br>
&gt; Apptainer have not been extensively tested yet across High-Performance-Computing environments. Hence, there may be issues.
</details>
