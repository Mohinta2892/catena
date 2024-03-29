## Download Public Datasets

## Organise your data for training and inference with [create_dir_organisation.py](create_dir_organisation.py)

- The script creates a directory for each domain in the base directory, with subdirectories for 2D and 3D data (`data_2d`, `data_3d`) and further subdivisions into `train` and `test` splits.
- A `preprocessed` directory will also be created within the base directory.
- Domain names are case-insensitive and will be converted to uppercase in the directory structure.

>[!WARNING]
> Ensure you are running this from the `lsd conda` env OR within `docker/apptainers` with python>=3.9 installed.

Run [create_dir_organisation.py](create_dir_organisation.py):
```
python /path/to/create_directory_structure.py </path/to/base_dir> <Brainvolname1> <Brainvolname2> ... <BrainvolnameN>

```
Run with --help to understand syntax better:
```
python /path/to/create_directory_structure.py --help
```
