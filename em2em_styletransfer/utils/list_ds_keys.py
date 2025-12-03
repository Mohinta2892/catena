import zarr
import h5py
from pathlib import Path


def list_keys(f):
    """
    List all keys in a zarr/hdf file
    Args:
        path: path to the zarr/hdf file

    Returns:
        list of keys
    """

    def _recursive_find_keys(f, base: Path = Path('/')):
        _list_keys = []
        for key, dataset in f.items():
            if isinstance(dataset, zarr.Group):
                new_base = base / key
                _list_keys += _recursive_find_keys(dataset, new_base)

            elif isinstance(dataset, h5py._hl.group.Group):
                new_base = base / key
                _list_keys += _recursive_find_keys(dataset, new_base)

            elif isinstance(dataset, zarr.Array):
                new_key = str(base / key)
                _list_keys.append(new_key)

            elif isinstance(dataset, h5py._hl.dataset.Dataset):
                new_key = str(base / key)
                _list_keys.append(new_key)

        return _list_keys

    return _recursive_find_keys(f)
