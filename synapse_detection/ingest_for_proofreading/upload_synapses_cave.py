import os
import sys
import logging
import json
import sqlite3
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from caveclient import CAVEclient
import argparse

# --- Configuration ---
DEFAULT_DB_PATH = "/path/to/your/synapse_predictions.db"
BATCH_SIZE = 10000
MAX_WORKERS = 8
STATE_FILE = "upload_state.json"


# --- Helper Functions ---

def get_db_connection(db_path):
    return sqlite3.connect(db_path)


def get_max_rowid(db_path):
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(rowid) FROM pre_post_mapping")
        result = cursor.fetchone()
        return int(result[0]) if result[0] is not None else 0


def get_remaining_count(db_path, start_id):
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pre_post_mapping WHERE rowid > ?", (start_id,))
        result = cursor.fetchone()
        return int(result[0]) if result[0] is not None else 0


def fetch_batch(db_path, start_id, limit):
    query = """
    SELECT 
        map.rowid AS id,
        pre.score,
        pre.x AS pre_x, pre.y AS pre_y, pre.z AS pre_z,
        post.x AS post_x, post.y AS post_y, post.z AS post_z
    FROM pre_post_mapping map
    JOIN pre_sites pre ON pre.id = map.pre_id
    JOIN post_sites post ON post.id = map.post_id
    WHERE map.rowid > ?
    ORDER BY map.rowid ASC
    LIMIT ?
    """
    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(start_id, limit))
    return df


def process_and_upload_chunk(chunk_data):
    """
    Worker function. Fetches its own batch from the DB to avoid
    pickling DataFrames (which re-introduces numpy int64 contamination).
    Receives only primitive types: db_path, start_id, limit, and connection params.
    """
    db_path = chunk_data['db_path']
    start_id = chunk_data['start_id']  # pure Python int
    limit = chunk_data['limit']
    server_address = chunk_data['server_address']
    datastack_name = chunk_data['datastack_name']
    token = chunk_data['token']
    table_name = chunk_data['table_name']

    # Fetch inside the worker — no DataFrame crosses the process boundary
    df = fetch_batch(db_path, start_id, limit)

    if df.empty:
        return 0, start_id

    client = CAVEclient(server_address=server_address,
                        datastack_name=datastack_name,
                        auth_token=token)
    ann = client.annotation

    # Build upload_df using pure Python ints throughout.
    # int() on numpy scalars gives a genuine Python int — safe for JSON.
    upload_df = pd.DataFrame({
        'id': [int(x) for x in df['id']],
        'pre_pt_position': [
            [int(r['pre_x']), int(r['pre_y']), int(r['pre_z'])]
            for _, r in df.iterrows()
        ],
        'post_pt_position': [
            [int(r['post_x']), int(r['post_y']), int(r['post_z'])]
            for _, r in df.iterrows()
        ],
    })
    # Keep id as plain Python object so it never gets cast back to int64
    upload_df['id'] = upload_df['id'].astype(object)

    batch_max_id = int(df['id'].max())  # pure Python int for safe state saving

    try:
        stage = ann.stage_annotations(table_name, id_field=True)
        stage.add_dataframe(upload_df)
        ann.upload_staged_annotations(stage)
    except Exception as e:
        logging.error(f"Upload failed for batch starting at {start_id}: {e}")
        raise e

    return len(upload_df), batch_max_id


# --- State Management ---

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Guard against a truncated / corrupt file
                if 'last_uploaded_id' in state:
                    return {'last_uploaded_id': int(state['last_uploaded_id'])}
        except (json.JSONDecodeError, ValueError):
            pass
    return {'last_uploaded_id': 0}


def save_state(last_id):
    # Always write a pure Python int so json.dump never sees numpy types
    with open(STATE_FILE, 'w') as f:
        json.dump({'last_uploaded_id': int(last_id)}, f)


def reset_state():
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Upload synapses to CAVE (Nanometer Mode).")
    parser.add_argument('datastack', help="The name of the CAVE datastack")
    parser.add_argument('--db', default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument('--reset', action='store_true', help="Reset progress AND delete existing annotations in CAVE")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument('--verify', action='store_true',
                        help="Upload ONLY the first 10,000 synapses to verify alignment.")

    args = parser.parse_args()

    db_path = args.db
    datastack_name = args.datastack
    token = os.environ.get('GOOGLE_SECRETS')

    destination_table_name = 'synapses_v7'
    destination_table_desc = 'Whole brain synapse prediction (Filtered, Nanometers)'

    client = CAVEclient(server_address='https://global.connectomics.braininbrain.org',
                        auth_token=token,
                        datastack_name=datastack_name)

    ann = client.annotation

    # Create table if missing
    try:
        ann.create_table(table_name=destination_table_name,
                         schema_name='nocleft_synapse',
                         voxel_resolution=[1, 1, 1],
                         description=destination_table_desc)
        print(f"Table '{destination_table_name}' created.")
    except Exception:
        print(f"Table '{destination_table_name}' already exists, continuing.")

    # --- COUNT AND DELETE LOGIC ---
    try:
        current_count = ann.get_annotation_count(destination_table_name)
        print(f"\n📊 Current annotations in '{destination_table_name}': {current_count}")

        if args.reset:
            logging.warning("Resetting local state...")
            reset_state()

            if current_count > 0:
                print(f"⚠️  Deleting {current_count} existing annotations from CAVE...")

                with get_db_connection(db_path) as conn:
                    df_ids = pd.read_sql_query("SELECT rowid FROM pre_post_mapping", conn)
                    all_ids = [int(x) for x in df_ids['rowid'].tolist()]

                for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="Deleting annotations"):
                    chunk_ids = all_ids[i:i + BATCH_SIZE]
                    try:
                        ann.delete_annotation(destination_table_name, chunk_ids)
                    except Exception:
                        pass  # Silently skip IDs that don't exist on the server

                new_count = ann.get_annotation_count(destination_table_name)
                print(f"✅ Deletion complete. New annotation count (Valid): {new_count}")

    except Exception as e:
        print(f"Could not retrieve or delete annotations: {e}")

    # --- State & Limits ---
    state = load_state()
    start_id = int(state['last_uploaded_id'])
    max_id = get_max_rowid(db_path)

    if args.verify:
        print("\n--- VERIFICATION MODE ---")
        max_processing_id = start_id + BATCH_SIZE
    else:
        max_processing_id = max_id

    if start_id >= max_id:
        print("Upload already complete.")
        sys.exit(0)

    total_rows = get_remaining_count(db_path, start_id)
    if args.verify:
        total_rows = min(total_rows, BATCH_SIZE)

    print(f"Resuming from rowid > {start_id}, processing up to rowid {max_processing_id} ({total_rows} synapses)")
    current_id_pointer = start_id

    # --- Worker Loop ---
    # Key design change: workers receive (db_path, start_id, limit) instead of a DataFrame.
    # This avoids pickling numpy arrays across process boundaries entirely.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        with tqdm(total=total_rows, unit="syn") as pbar:

            while current_id_pointer < max_processing_id:
                futures = {}
                tasks_to_submit = 1 if args.verify else args.workers * 2

                # Pre-compute batch start IDs by peeking at rowids
                for _ in range(tasks_to_submit):
                    if current_id_pointer >= max_processing_id:
                        break

                    # Peek at the next batch to find its max rowid so we can
                    # advance the pointer correctly without fetching data twice.
                    # We do a lightweight rowid-only query here.
                    with get_db_connection(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT MAX(rowid) FROM ("
                            "  SELECT rowid FROM pre_post_mapping"
                            "  WHERE rowid > ? ORDER BY rowid ASC LIMIT ?"
                            ")",
                            (current_id_pointer, BATCH_SIZE)
                        )
                        result = cursor.fetchone()

                    if result[0] is None:
                        current_id_pointer = max_processing_id
                        break

                    batch_max_id = int(result[0])

                    task_args = {
                        'db_path': db_path,
                        'start_id': current_id_pointer,  # pure Python int
                        'limit': BATCH_SIZE,
                        'server_address': client.server_address,
                        'datastack_name': datastack_name,
                        'token': token,
                        'table_name': destination_table_name,
                    }

                    future = executor.submit(process_and_upload_chunk, task_args)
                    futures[future] = batch_max_id

                    current_id_pointer = batch_max_id

                for future in as_completed(futures):
                    batch_max_id = futures[future]
                    try:
                        uploaded_count, actual_max_id = future.result()
                        pbar.update(uploaded_count)
                        save_state(actual_max_id)  # always a pure Python int from the worker
                    except Exception as e:
                        logging.error(f"Batch ending at {batch_max_id} failed: {e}")
                        # Do NOT save state for this batch — it will be retried on resume

                if args.verify:
                    print("\n✅ Verification batch uploaded.")
                    break

    if not args.verify:
        print("\nUpload complete. Ingesting...")
        try:
            client.materialize.ingest_annotation_table(destination_table_name)
            print("✅ Ingestion triggered successfully.")
        except Exception as e:
            print(f"Ingest failed: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
