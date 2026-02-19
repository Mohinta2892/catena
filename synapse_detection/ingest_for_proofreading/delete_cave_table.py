import os
import sys
from caveclient import CAVEclient

# CONFIG
DATASTACK_NAME = 'zlatic_octo_8x8x8_full_200525_datastack'
TABLE_NAME = 'synapses_v2'
TOKEN = os.environ.get('GOOGLE_SECRETS')

client = CAVEclient(server_address='https://global.connectomics.braininbrain.org',
                    auth_token=TOKEN,
                    datastack_name=DATASTACK_NAME)

print(f"⚠️  DELETING TABLE: {TABLE_NAME} ...")
try:
    client.annotation.delete_table(TABLE_NAME)
    print("✅ Table deleted successfully.")
except Exception as e:
    print(f"❌ Could not delete table (it might not exist or you lack permission): {e}")
