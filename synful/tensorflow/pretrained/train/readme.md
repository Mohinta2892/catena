##  Quick glance: Pretrained model specs
Credits: Julia Buhmann

| Setup                     | Specs                           | F-score with seg | F-score without | Remarks|
|---------------------------|---------------------------------|------------------|------------------|------------------|
| `p_setup52` (`+`p_setup10)    | big, curriculum, CE, ST          | 0.76             | 0.74             | **Highest Accuracy** |
| `p_setup51`                   | big, curriculum, CE, MT_2       | 0.76             | 0.73             |
| `p_setup54` (`+`p_setup05)    | small, curriculum, MSE, ST       | 0.76             | 0.7              | **Fast inference, reasonable results** |
| `p_setup45` (`+`p_setup05)    | small, standard, MSE, MT2        | 0.73             | 0.68             |

Note, that for the models that have an underlying ST architecture we also indicate the setup for the corresponding direction-vector-models (`p_setup05+p_setup10`). If you want to use the model with highest accuracy, pick `p_setup52`. If you want to use a model that gives reasonnable results, but also has fast inference runtime, pick `p_setup54`.

<details><summary>Notations</summary>
<br>

- MT 1: Multi-headed UNET with multi-task learning of post-synaptic masks and pre-synaptic direction vectors.
- MT 2: Independent upsampling paths in UNET for multi-task learning of post-synaptic masks and pre-synaptic direction vectors.
- ST: Single-headed UNET to learn either post-synaptic masks or pre-synaptic direction vectors.
- CE: Trained with Cross-Entropy Loss
- MSE: Trained with Mean Squared Error Loss

</details>


## Download Checkpoints
Checkpoints: [Link](https://www.dropbox.com/scl/fo/hlw1cbef09xwisss59fhr/h?rlkey=uk7786539u1fu21dh5ebowip6&dl=0)

These checkpoints are from training on [CREMI-realigned datasets](https://github.com/funkelab/synful/tree/master?tab=readme-ov-file).

> [!IMPORTANT]
> INSTALL [Docker](https://docs.docker.com/engine/install/) and [MongoDB](https://www.mongodb.com/docs/manual/installation/).
> Setup [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) for GPU acceleration.

## Pull docker image
```bash

docker pull mohinta2892/synful_tf1_py3:latest

```

## Data/Code Organisation
Please follow the below steps to be able to run the pretrained models on your own data:

- git clone this repo
- `cd synful/tensorflow/pretrained/train`
- `mkdir code data`
- Copy the respective `checkpoints` to their corresponding `setup` folders.
<br> Example organisation:
    ```bash
    - pretrained
        - train
            - code
                - p_setup10
                    - predict.py
                    - predict_blockwise.py
                    - train_net.meta
                    - train_net_checkpoint_390000.data-00000-of-00001
                    - train_net_checkpoint_390000.index
                    - train_net_checkpoint_690000.data-00000-of-00001
                    - train_net_checkpoint_690000.index
                    - train_net_config.json
                    - p_setup10_config.json
                    - extract_parameters_setup32.json
    ```
## Run inference

1. Check MongoDB is running:
  ```bash
  nc -zvv localhost 27017
  ```
>> Should Return: Connection to localhost (127.0.0.1) 27017 port [tcp/*] succeeded!
2. Check the docker image name:
  ```
  docker images
  ```
3. Run the loaded docker image:
  ```bash
  nvidia-docker run --shm-size 128gb --pids-limit -1 -it -u `id -u`:`id -g` -v `pwd`:`pwd` -w `pwd` -v {/path/to}/synful/tensorflow/:/home --network=host {nvcr.io/nvidia/tensorflow:21.12-tf1-py3}
  ```
  <details close>
   <summary>Command Explained Here</a></summary>
  <br>
  nvidia-docker run: Launches a new container using NVIDIA Docker, which allows the container to access NVIDIA GPUs.
  --shm-size 128gb: Sets the size of /dev/shm (shared memory) to 128GB.
  --pids-limit -1: Removes the limit on the number of processes that can be created inside the container.
  -it: Runs the container in interactive mode with a pseudo-TTY.
  -u id -u:id -g``: Sets the user and group IDs inside the container to match those of the current user on the host.
  -v pwd:pwd``: Mounts the current working directory from the host into the container at the same path.
  -w pwd``: Sets the working directory inside the container to match the current working directory on the host.
  -v {/path/to}/synful/tensorflow/:/home: Mounts a specific directory from the host into the /home directory in the container.
  --network=host: Uses the host's network stack inside the container.
  {nvcr.io/nvidia/tensorflow:21.12-tf1-py3}: Specifies the Docker image to use, which is an NVIDIA TensorFlow image version 21.12 with TensorFlow 1.x and Python 3.
  </details>

4. Change paths to `/home` inside the container. Check you can see all your files there.
  ```bash
  cd /home
  ls -lrt
  ```
  >> Should show something like:
  ```
      - pretrained
        - train
          - code
          - data
  ```
5. Based on your requirements and the above the `model performance` note, choose a `setup`.
6. If you choose `p_setup05`, edit the `p_setup05_config.json` 
<br>
<div>
<p align="center">
<img src='https://github.com/Mohinta2892/catena/blob/dev/synful/assets/synful_pretrained_anatomy.png' align="center" width=400px>
</p>
</div>

> [!NOTE]
> We have made changes to the `predict_blockwise.py` and `predict.py` files. Check [CHANGELOG.md](https://github.com/Mohinta2892/catena/blob/dev/synful/tensorflow/pretrained/train/CHANGELOG.md).

 7. Run inference:
   ```bash
      python predict_blockwise.py p_setup05_config.json
   ```
  A successful run outputs information on screen ([Example log](https://github.com/Mohinta2892/catena/blob/dev/synful/tensorflow/pretrained/train/p_setup05/example_log.txt)).

8. Output saved as zarr under `./output/p_setup05/{output_filename_as_in_config.zarr}`.
9. Further metadata in MongoDB database, these are needed during synapse extraction.
