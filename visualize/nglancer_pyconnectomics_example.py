import neuroglancer
import numpy as np
import argparse
import h5py
from funlib.persistence import open_ds
from funlib.geometry import Roi
from collections import OrderedDict
import dask.array as da

ip = '0.0.0.0'  # or public IP of the machine for sharable display
port = 9999  # change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
viewer = neuroglancer.Viewer()


# OCTO (# 3d vol dim: z,y,x)
def set_coordinate_space(names=['z', 'y', 'x'], units=['nm', 'nm', 'nm'], scales=[8, 8, 8]):
    res = neuroglancer.CoordinateSpace(
        names=names,
        units=units,
        scales=scales)
    return res


res0 = set_coordinate_space(names=['z', 'y', 'x'],
                            units=['nm', 'nm', 'nm'],
                            scales=[8, 8, 8])

# coordinate space for AFF volume (c,z,y,x)
res1 = set_coordinate_space(
    names=['c^', 'z', 'y', 'x'],
    units=['', 'nm', 'nm', 'nm'],
    scales=[1, 8, 8, 8])

raw_file = "/media/samia/DATA/ark/lsd_outputs/AFF/3d/run-aclsd-together/segmented/crop_A1_z16655-17216_y13231-13903_x7650-8468.zarr"
raw = open_ds(raw_file, "volumes/raw").data
seg = open_ds(raw_file, "volumes/segmentation_055").data
aff = open_ds(raw_file, "volumes/pred_affs").data
d_seg = da.from_array(seg, chunks='auto')
d_raw = da.from_array(raw, chunks='auto')
d_aff = da.from_array(aff, chunks='auto')
print(d_seg.T)
unique_equivalences = da.unique(d_seg)


# # to render 3D meshes, we have to copy this to the segmentation layer in the viewer
# print(f"To see 3D meshes of all segments copy this to the segmentation tab:"
#       f"{np.unique(mtlsd_seg)}")

def ngLayer(data, res, oo=[0, 0, 0], tt='segmentation'):
    return neuroglancer.LocalVolume(data, dimensions=res, volume_type=tt, voxel_offset=oo)


with viewer.txn() as s:
    s.layers.append(name='image', layer=ngLayer(d_raw, res0, tt='image'))
    s.layers.append(name='segmentation', layer=ngLayer(d_seg, res0, tt='segmentation'))
    s.layers.append(name='affinities', layer=ngLayer(d_aff, res1, oo=[0, 0, 0, 0], tt='image'),
                    shader="""
        void main() {
        emitRGB(vec3(toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue(2))));
        }
    """)
    s.selectedLayer.layer = "segmentation"
    s.selectedLayer.visible = True
    s.layers["segmentation"].tab = "segments"
    s.selectedLayer.size = 500  # sets the width of the right hand panel

    s.layers["segmentation"].skeletonRendering = \
        OrderedDict([('mode2d', 'lines_and_points'), ('mode3d', 'lines')])
    s.layers["segmentation"].segments = unique_equivalences.compute()  # compute this dask array here

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--static-content-url')
    ap.add_argument('-a', '--bind-address', default="localhost")
    args = ap.parse_args()
    neuroglancer.server.debug = True
    if args.bind_address:
        neuroglancer.set_server_bind_address(args.bind_address)
    if args.static_content_url:
        neuroglancer.set_static_content_source(url=args.static_content_url)
    print(viewer)
