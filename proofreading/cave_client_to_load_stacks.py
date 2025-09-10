"""
This script is a first trial at trying to talk to the FlyWire Octo instance via the CAVEclient.
Ultimately, we would want to programmatically merged catmaid tracings via the caveclient.

Run (in beast): conda deactivate; conda activate ngl

Author: Samia Mohinta
Affiliation: Cambridge University, UK
"""

from caveclient import CAVEclient

server_addrs = "https://global.connectomics.braininbrain.org"
datastack = "zlatic_octo_8x8x8_full_200525_datastack"

client = CAVEclient(server_address=server_addrs)
auth = client.auth
current_token = auth.token
print(f"My current token is: {auth.token}")

# # we need to get a token to make the connection
# print(auth.get_new_token()) # this should show a message

new_token = 'f6721f14bd4a5ee68df9f725fdb47c92'  # This is the text you see after you visit the website.
try:
    if new_token != current_token:
        auth.save_token(token=new_token)  # save to disk
        print(f"My token is now: {auth.token}")

except Exception as e:
    pass

# Prints like: ['zlatic_octo_8x8x8_datastack', 'zlatic_sam3g_090625_rsg32_datastack', 'zlatic_octo_8x8x8_full_200525_datastack']
print(client.info.get_datastacks())

# Choose a datastack to initialize with
datastack_name = 'zlatic_octo_8x8x8_full_200525_datastack'
client = CAVEclient(datastack_name=datastack_name, server_address=server_addrs, auth_token=new_token)
# #
print(f"versions: {client.materialize.get_versions()}")
#
for version in client.materialize.get_versions():
    print(f"Version {version}: {client.materialize.get_timestamp(version)}")

# Try to find all leaves for a root-id
root_id = 648518346360884000 #648518346363224861
# there are max of 7 levels. higher you go, the number of leaves decreases
leaves = client.chunkedgraph.get_leaves(root_id, stop_layer=None)
print(f"leaves: {leaves}, len of leaves {len(leaves)}")

root_ids = [648518346360884739, 648518346370540737, 648518346364422034, 648518346360360026, 648518346362705549, 648518346364413735, 648518346363644529, 648518346362749310, 648518346362801892, 648518346367787881, 648518346362429739, 648518346363695060]


print(f'change log: {client.chunkedgraph.get_change_log(root_id, filtered=False)}')
print(f'merge log: {client.chunkedgraph.get_merge_log(root_id)}')
print(f'tabular log: {client.chunkedgraph.get_tabular_change_log(root_id,filtered=False)}')
print(f'original roots: {client.chunkedgraph.get_original_roots(root_id)}')
print('timestamps')
for id_ in root_ids:
    print(f'{client.chunkedgraph.get_root_timestamps(id_)}')

### Datetime analysis

import datetime
import pytz

# 1. Your initial data, including empty lists.
timestamps_with_empties = [
    [datetime.datetime(2025, 6, 9, 10, 58, 0, 689000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 17, 14, 54, 14, 389000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 18, 12, 37, 9, 699000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 18, 15, 54, 26, 448000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 19, 11, 39, 0, 893000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 19, 14, 48, 30, 753000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 20, 8, 32, 7, 641000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 20, 10, 17, 1, 860000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 20, 14, 55, 39, 424000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 2, 9, 11, 21, 68000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 22, 20, 40, 55, 819000, tzinfo=pytz.UTC)],
    [datetime.datetime(2025, 6, 23, 7, 57, 28, 453000, tzinfo=pytz.UTC)],
]

# 2. Filter out the empty lists and extract the datetime objects.
# This list comprehension iterates through the outer list, and for each non-empty
# inner list, it takes the first (and only) datetime object.
valid_timestamps = [item[0] for item in timestamps_with_empties if item]

# 3. Find the most recent (maximum) datetime object from the filtered list.
latest_datetime = max(valid_timestamps)

# 4. Convert the latest datetime object to a Unix timestamp and cast to an integer.
unix_timestamp = int(latest_datetime.timestamp())

# unix_timestamp = int(datetime.datetime(2025, 6, 21, 14, 55, 39, 424000, tzinfo=pytz.UTC).timestamp())

print(f"Latest datetime object: {latest_datetime}")
print(f"Converted Unix timestamp: {unix_timestamp}")

# --- Verification ---
# Latest datetime object: 2025-06-20 14:55:39.424000+00:00
# Converted Unix timestamp: 1750431339


input_leaves = [150994947, 1216348164]
root_from_leaves = client.chunkedgraph.get_roots(input_leaves)
print(f"root for leaves {root_from_leaves}")

# bounds of a root_id
root_info = client.chunkedgraph.get_roots([root_id], bounds=True)
source_bbox = root_info[0]['bbox_mip0']
print(source_bbox)

# nodes = client.chunkedgraph.get_minimal_covering_nodes([root_id])
# print(f"nodes from rootid {nodes}")

# find the base segmentation
# print(f"base segmentation {client.chunkedgraph.segmentation_info}")

# check any associated tables.
"""# currently throws: requests.exceptions.HTTPError: 500 Server Error: datastack zlatic_octo_8x8x8_full_200525_datastack info 
# not returned 403 Client Error:
# FORBIDDEN for url: https://global.connectomics.braininbrain.org/info/api/v2/datastack/full/zlatic_octo_8x8x8_full_200525_datastack
# content: b'{\n  "data": {\n    "auth_dataset": "zlatic_octo_8x8x8_full200525",\n    "required_permission": "view"\n  },\n  
# "error": "missing_permission",\n  
# "message": "Missing permission: view for dataset zlatic_octo_8x8x8_full200525"\n}\n'
# for url: https://local.cave.braininbrain.org/materialize/api/v3/datastack/zlatic_octo_8x8x8_full_200525_datastack/versions?expired=False 
"""
# print(client.materialize.get_tables(datastack_name=datastack_name))

# Try to load a skids but requires an L2 cache which seems like something to already have in the PCG?!
# print(f"skeleton version {client.skeleton.get_version()}")
# example_cell_id = 648518346363224861
# sk_df = client.skeleton.get_skeleton(example_cell_id, output_format='swc')
# print(sk_df.head())

# def calc_area_vol_cells(root_id):
#     import pandas as pd
#     lvl2nodes = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
#     l2stats = client.l2cache.get_l2data(lvl2nodes, attributes=['size_nm3', 'area_nm2'])
#     l2df = pd.DataFrame(l2stats).T
#     total_area_um2 = l2df.area_nm2.sum() / (1000 * 1000)
#     total_volume_um3 = l2df.size_nm3.sum() / (1000 * 1000 * 1000)
#     print(f"volume um3: {total_volume_um3}, area um2:  {total_area_um2}")
#
#
# calc_area_vol_cells(root_id)

# try to find the lineage tree
# proofread_neuron = 648518346360884739
# print(client.chunkedgraph.get_lineage_graph(proofread_neuron))

# This works: the point coords must be in pixels to get ids
import pandas as pd

df = pd.read_csv(
    "/media/samia/DATA/mounts/fibserver1/smohinta_data/proofreading/tracedCATMAID_neuroglancer/octo_229050_nodes.csv")
resolution = (8, 8, 8)
x, y, z = df["x"] // resolution[0], df["y"] // resolution[1], df["z"] // resolution[2]
all_node_locs = [(int(xi), int(yi), int(zi)) for xi, yi, zi in zip(x, y, z)]
# client = CAVEclient(datastack_name)
cv = client.info.segmentation_cloudvolume()
pts = cv.scattered_points(
    all_node_locs,  # points to look up
    coord_resolution=cv.meta.resolution(0),  # resolution those points are specified in, here I specify voxel space
    agglomerate=True  # lookup supervoxels, if you want root set agglomerate=True but this adds overhead
)
print(pts)

df = pd.DataFrame.from_dict(pts)
df.to_csv("/media/samia/DATA/mounts/fibserver1/smohinta_data/proofreading/octo_229050_nodes_flywire.csv", index=False)

pts_id = pts.values()
print(pts_id)

# skel = cv.skeleton.get(pts_id)
mesh = cv.mesh.get(pts_id)  # return the mesh as vertices and faces instead of writing to disk

print(mesh)
