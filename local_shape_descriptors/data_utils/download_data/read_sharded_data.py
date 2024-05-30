import cloudvolume
from cloudvolume import CloudVolume
import numpy as np
import argparse

import zarr


def write_to_zarr(outfile, data, resolution, transpose=True, offset=(0, 0, 0)):
    file_ = zarr.open(outfile, "a")

    # per our convention we should save the data as zyx
    if transpose:
        data = np.transpose(data, (2, 1, 0))
        # resolution = np.transpose(np.array(resolution.data), (2, 1, 0))

    file_["volumes/raw"] = data
    file_["volumes/raw"].attrs["resolution"] = resolution
    file_["volumes/raw"].attrs["offset"] = offset

    print(f" Saved here: {outfile}")


def resolution_tuple(resolution_str):
    # Custom function to parse a resolution string into a tuple of integers

    try:
        if isinstance(resolution_str, list) and len(resolution_str) == 1:
            resolution_str = resolution_str[0]
        # else:
        #     raise argparse.ArgumentTypeError(f"Passed arg cannot be parsed, it is like {resolution_str}")
        parts = resolution_str.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Resolution must contain exactly three integers separated by commas.")
        return tuple(int(part.strip()) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Resolution must contain only integers separated by commas.")


def read_shards_as_cloudvolume(filename, args, bbox_start=(0, 0, 0), bbox_end=(1024, 1024, 1024), mip=1):
    vol = CloudVolume(f"precomputed://file://{filename}", fill_missing=True)
    # volume statistics
    print(f"volume info {vol.info}")
    print(f"volume shape {vol.shape}")
    print(f"volume grid size {vol.image.grid_size()}")

    # vol.dataset_name
    # vol.image
    # vol.image.grid_size()
    # vol.image.has_data(mip=1)  # mip = resolution scale

    # this bbox shape in xyz must be known prior - could be the  whole shape at mip level or a ROI
    # e.g. mipp=1 whole grid leonardo (2264, 2597, 1859)
    bbox = cloudvolume.Bbox(bbox_start, bbox_end)  # this is in xyz

    # files object will contain the ROI/whole volume at give mip level
    files = vol.download(bbox, mip=mip)

    # grab the data
    # data = files.data
    data = np.squeeze(files.data)  # contains channel dim, hence squeeze

    resolution = files.resolution # a Vec() object use this later

    write_to_zarr(outfile=args.of, data=data, resolution=resolution_tuple(args.res), transpose=args.trans,
                  offset=resolution_tuple(args.offset)
                  )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        default='/media/samia/DATA/ark/dan-samia/lsd/funke/neptune/google_CLAHED_EM/8x8x8_cropped_clahe',
                        help="Input the neuroglancer multiscale file")
    parser.add_argument('-of',
                        default="/media/samia/DATA/ark/dan-samia/lsd/funke/neptune/zarr/neptune_zyx_32_32_32.zarr",
                        help="Provide the full path to the output zarr filename")
    parser.add_argument('-mip', default=0, type=int, help="Pass the scale level you want to save as zarr")
    parser.add_argument('-trans', default=True, type=bool,
                        help="Pass True if you want to transpose data from xyz to zyx."
                             " Original data can be in xyz.")
    parser.add_argument('-offset', nargs='+', default="0,0,0", help="Pass an offset to the data")
    parser.add_argument('-res', nargs='+', default="8, 8, 8", help="Pass data resolution as Z,Y,X")
    parser.add_argument('-bbox_start', nargs='+', default="0,0,0", help="Pass boundary box start like X,Y,Z to the data")
    parser.add_argument('-bbox_end', nargs='+', default="4416,2912,2848", help="Pass boundary box end like"
                                                                                 " X,Y,Z to the data."
                                                                                 " Check info file in the "
                                                                                 "gcloud dataset to get full "
                                                                                 "dimensions info at every scale")

    args = parser.parse_args()

    read_shards_as_cloudvolume(args.f, args, bbox_start=resolution_tuple(args.bbox_start),
                               bbox_end=resolution_tuple(args.bbox_end), mip=args.mip)


if __name__ == '__main__':
    main()
