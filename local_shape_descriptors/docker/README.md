# LSD Docker & Apptainer Installation

### Docker
1. Install [Docker Engine](https://docs.docker.com/engine/install/) and verify installation in your OS environment.
2. Running docker images:
  1. You can pull the latest docker image from DockerHub:
      ```
      docker pull mohinta2892/lsd_sheridan_pytorch:22.01-py3
      ```
     This will load the image onto your local memory.
  3. You can build the docker image yourself:
     ``` 
      cd catena/local_shape_descriptors/docker
      docker build -t lsd_sheridan_pytorch:latest .
     ```
        -t tag the docker image with a name (lsd_sheridan_pytorch) and version (latest).
> [!TIP]
> You can pass Dockerfiles to docker build as ` docker build -f /path/to/Dockerfile.updated /catena/local_shape_descriptors `
> `/catena/local_shape_descriptors` is the context given to the docker. Meaning of [commands](https://docs.docker.com/reference/cli/docker/image/build/)
