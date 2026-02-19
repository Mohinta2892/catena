import sqlite3
import pandas as pd
import os

# Update path if necessary
DB_PATH = "/mnt/graid/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000_filt/final_upload_ready.db"

def inspect_database():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    print(f"--- Database Inspection: {os.path.basename(DB_PATH)} ---\n")

    # 1. Count Rows
    tables = ['pre_sites', 'post_sites', 'pre_post_mapping']
    for table in tables:
        count = pd.read_sql_query(f"SELECT count(*) FROM {table}", conn).iloc[0,0]
        print(f"Table '{table}': {count:,} rows")

    print("\n--- Sample Data (Joined) ---")
    
    # 2. Join tables to show full context (Coordinates + Seg IDs + Score)
    query = """
    SELECT 
        map.pre_id, 
        pre.score,
        pre.segment_id AS pre_seg,
        post.segment_id AS post_seg,
        pre.x AS pre_x, pre.y AS pre_y, pre.z AS pre_z,
        post.x AS post_x, post.y AS post_y, post.z AS post_z
    FROM pre_post_mapping map
    JOIN pre_sites pre ON map.pre_id = pre.id
    JOIN post_sites post ON map.post_id = post.id
    LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    # 3. Validation Checks
    print("\n--- Integrity Checks ---")
    
    # Check for Autapses
    autapses = pd.read_sql_query(
        "SELECT count(*) FROM pre_post_mapping WHERE pre_seg_id = post_seg_id", conn
    ).iloc[0,0]
    
    if autapses == 0:
        print("✅ Autapse Check Passed: 0 autapses found.")
    else:
        print(f"❌ Autapse Check Failed: {autapses} autapses found!")

    # Check for Zero Segments (Background)
    zeros = pd.read_sql_query(
        "SELECT count(*) FROM pre_post_mapping WHERE pre_seg_id = 0 OR post_seg_id = 0", conn
    ).iloc[0,0]
    
    if zeros == 0:
        print("✅ Background Check Passed: No synapses mapped to segment 0.")
    else:
        print(f"❌ Background Check Failed: {zeros} synapses mapped to background (0).")

    conn.close()

if __name__ == "__main__":
    inspect_database()
