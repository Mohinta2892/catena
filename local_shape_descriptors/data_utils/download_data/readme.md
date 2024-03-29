## Download Public Datasets [download_volumes.py](download_volumes.py)

This Python script automates the download of specific datasets from an AWS S3 bucket. The script reads dataset configurations from a `datasets.json` file and prepares download jobs for specified data volumes within a given S3 bucket path. Utilizing multiprocessing, it concurrently downloads these datasets into a local directory structure that mirrors the S3 bucket's organization.

>[!IMPORTANT]
>Follow these instructions to setup AWS Credentials. You need them to download the datasets.
>To configure your AWS access and secret key on a bash shell run:
>```
>aws configure # creds get stored here:  ~/.aws/credentials
>```
>Add the following lines to your `~/.bash_profile`:
>```
>export AWS_SECRET_ACCESS_KEY='your_secret_access_key_here'
>export AWS_ACCESS_KEY_ID='your_access_key_id_here'
>```
>Then, apply the changes:
>```
>source ~/.bash_profile
>```

Run [download_volumes.py](download_volumes.py):
```
cd catena/local_shape_descriptors/data_utils/download_data
python download_volumes.py
```
>[!Note]
> Remember you need access to `datasets.json`, so run it from within `catena/local_shape_descriptors/data_utils/download_data`

## Organise your data for training and inference with [create_dir_organisation.py](create_dir_organisation.py)

- The script creates a directory for each domain in the base directory, with subdirectories for 2D and 3D data (`data_2d`, `data_3d`) and further subdivisions into `train` and `test` splits.
- A `preprocessed` directory will also be created within the base directory.
- Domain names are case-insensitive and will be converted to uppercase in the directory structure.

>[!WARNING]
> Ensure you are running this from the `lsd conda` env OR within `docker/apptainers` with python>=3.8 and python<3.11 installed.
> python>=3.11 may cause unforeseen problems. We will update the documentation once we test the codebase on these versions.

Run [create_dir_organisation.py](create_dir_organisation.py):
```
python /path/to/create_directory_structure.py </path/to/base_dir> <Brainvolname1> <Brainvolname2> ... <BrainvolnameN>

```
Run with --help to understand syntax better:
```
python /path/to/create_directory_structure.py --help
```
