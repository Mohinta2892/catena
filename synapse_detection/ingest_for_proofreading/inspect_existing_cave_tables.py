import os
from caveclient import CAVEclient
import pandas as pd

# CONFIG
DATASTACK_NAME = 'zlatic_octo_8x8x8_full_200525_datastack' # Update to your actual datastack
TABLE_NAME = 'synapses_v3'
TOKEN = os.environ.get('GOOGLE_SECRETS') # Or paste your token string

client = CAVEclient(server_address='https://global.connectomics.braininbrain.org',
                    auth_token=TOKEN,
                    datastack_name=DATASTACK_NAME)

# 1. Check if table exists
all_tables = client.annotation.get_tables()
if TABLE_NAME not in all_tables:
    print(f"Table '{TABLE_NAME}' does NOT exist.")
else:
    print(f"✅ Table '{TABLE_NAME}' exists.")

    # 2. Get Metadata (Owner, Description, Schema)
    meta = client.annotation.get_table_metadata(TABLE_NAME)
    
    print("\n--- Table Metadata ---")
    print(f"User ID:     {meta.get('user_id')}")
    print(f"Description: {meta.get('description')}")
    print(f"Schema:      {meta.get('schema_type')}")
    print(f"Resolution:  {meta.get('voxel_resolution')}")
    print(f"Row Count:   {meta.get('max_annotation_id')}")
    
    # 3. Check Permissions
    print(f"Read Perms:  {meta.get('read_permission')}")
    print(f"Write Perms: {meta.get('write_permission')}")
    
    # delete table as specified in table_name
    #client.annotation.delete_table(TABLE_NAME)
