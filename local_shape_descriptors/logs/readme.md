An example train_logs.txt is attached here. Each file will have a `datetime` in the filename to give an idea of the run.

The `logs` directory will be automatically created under `catena/local_shape_descriptors` when `trainer.py` or `predicter.py` is run.

>[!WARNING]
> Train Log files when saved on disk can be quite large (*> 500MB*), because models run for >=300000 epochs.
> We suggest you use bash command `tail`| `head` to peek inside these files.
> We will look into compressing these in the future.
