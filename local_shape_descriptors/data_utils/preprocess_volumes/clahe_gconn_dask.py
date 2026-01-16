import os
import dask.array
from dask import delayed
from dask.distributed import Client, LocalCluster
import skimage
import numpy as np
import tifffile
import zarr
from pathlib import Path
from tqdm.dask import TqdmCallback


class CLAHE:
    def __init__(
            self,
            kernel_size=None,
            clip_limit=0.01,
            clip_min=None,
            clip_max=None,
            invert=False
    ):
        self._kernel_size = kernel_size
        self._clip_limit = clip_limit
        self._invert = invert
        self._clip_max = clip_max
        self._clip_min = clip_min

    def process_plane(self, image2d: np.ndarray) -> np.ndarray:
        if len(set(np.unique(image2d))) == 1:
            return image2d

        if self._clip_min is not None or self._clip_max is not None:
            c_min = self._clip_min if self._clip_min is not None else -np.inf
            c_max = self._clip_max if self._clip_max is not None else np.inf
            image2d = np.clip(image2d, c_min, c_max)

        clahed = skimage.exposure.equalize_adapthist(
            image2d, kernel_size=self._kernel_size, clip_limit=self._clip_limit
        )
        if self._invert:
            clahed = 1.0 - clahed
        return (clahed * 255).astype(np.uint8)


def apply_clahe_to_chunk(chunk, clahe_params):
    apply_clahe = CLAHE(**clahe_params)
    return apply_clahe.process_plane(chunk)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='Input Zarr/tiff file')
    parser.add_argument('-of', default=None, help='Output Zarr/tiff file')
    parser.add_argument('-ds', default='volumes/raw', help='Dataset inside Zarr')
    parser.add_argument('-ods', default='volumes/raw', help='Output dataset in Zarr')
    parser.add_argument('-k', nargs='+', default="30 300 300",
                        help='Kernel size for CLAHE')
    parser.add_argument('-cl', default=0.01, help='Clip limit')
    parser.add_argument('-cmin', default=None, help='Clip minimum')
    parser.add_argument('-cmax', default=None, help='Clip maximum')
    parser.add_argument('-inv', default=False, help='Invert the CLAHE result')
    args = parser.parse_args()

    kernel_size = tuple(map(int, args.k))
    clahe_params = {
        'kernel_size': kernel_size,
        'clip_limit': float(args.cl),
        'clip_min': float(args.cmin) if args.cmin else None,
        'clip_max': float(args.cmax) if args.cmax else None,
        'invert': bool(args.inv),
    }

    if args.f.endswith(('.tif', '.tiff')):
        file_ = tifffile.imread(args.f, aszarr=True)
    elif args.f.endswith('.zarr'):
        file_ = zarr.open(args.f, mode='r')[args.ds]
    else:
        raise ValueError("Unsupported file type. Use Zarr or TIFF.")

    chunk_size = (1, 256, 256)
    data = dask.array.from_zarr(args.f, component=args.ds, chunks=chunk_size)

    if args.of is None:
        outfile_name = f"{os.path.splitext(args.f)[0]}_clahed.zarr"
    else:
        outfile_name = args.of

    out_zarr = zarr.open(outfile_name, mode='w')
    out_zarr.create_dataset(
        name=args.ods, shape=data.shape, dtype=np.uint8, chunks=chunk_size
    )

    # Setup Dask distributed cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    print(client.dashboard_link)  # Optional: Prints cluster details


    def process_and_save_chunk(chunk, block_info=None):
        block_id = block_info[None]['chunk-location']
        processed_chunk = apply_clahe_to_chunk(chunk, clahe_params)
        z, y, x = block_id
        out_zarr[args.ods][z:z + 1, y:y + chunk.shape[1], x:x + chunk.shape[2]] = processed_chunk
        return processed_chunk


    processed_chunks = data.map_blocks(process_and_save_chunk, dtype=np.uint8)

    with TqdmCallback(desc="Processing chunks"):
        processed_chunks.compute()
