"""
This script is a first trial at trying to talk to the FlyWire Octo instance via the CAVEclient.
Ultimately, we would want to programmatically merged catmaid tracings via the caveclient.

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

new_token = 'x'  # This is the text you see after you visit the website.
if new_token != current_token:
    auth.save_token(token=new_token)  # save to disk
    print(f"My token is now: {auth.token}")

# Prints like: ['zlatic_octo_8x8x8_datastack', 'zlatic_sam3g_090625_rsg32_datastack', 'zlatic_octo_8x8x8_full_200525_datastack']
print(client.info.get_datastacks())

# Choose a datastack to initialize with
datastack_name = 'zlatic_octo_8x8x8_full_200525_datastack'
client = CAVEclient(datastack_name=datastack_name)

# Try to find all leaves for a root-id
root_id = 648518346363224861
# there are max of 7 levels. higher you go, the number of leaves decreases
leaves = client.chunkedgraph.get_leaves(root_id, stop_layer=None)
print(f"leaves: {leaves}, len of leaves {len(leaves)}")

input_leaves = [150994947, 1216348164]
root_from_leaves = client.chunkedgraph.get_roots(input_leaves)
print(f"root for leaves {root_from_leaves}")

nodes = client.chunkedgraph.get_minimal_covering_nodes([root_id])
print(f"nodes from rootid {nodes}")

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
