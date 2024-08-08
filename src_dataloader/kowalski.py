import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from penquins import Kowalski
import multiprocessing
import dotenv

dotenv.load_dotenv()

# Constants
BASE_PATH = 'data_kowalski/'
DATA_PATH = 'BTS.csv'
BATCH_SIZE = 16
KOWALSKI_HOST = 'kowalski.caltech.edu'
KOWALSKI_PORT = 443

def ensure_directory_exists(path):
    """Ensure the base directory exists."""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Created base directory: {path}")
        except PermissionError as e:
            print(f"Permission denied: {e}")
            exit()

def get_kowalski_instance():
    """Initialize and return a Kowalski instance."""
    kowalski_token = os.getenv("KOWALSKI_API_TOKEN")
    instances = {
        'kowalski': {
            'protocol': 'https',
            'port': KOWALSKI_PORT,
            'host': KOWALSKI_HOST,
            'token': kowalski_token
        }
    }
    return Kowalski(instances=instances)

def check_kowalski_connection(k):
    """Check the connection to Kowalski."""
    if k.ping(name="kowalski"):
        print("Connected to Kowalski")
    else:
        print("Unable to connect to Kowalski")
        exit()

def load_bts_data(data_path):
    """Load BTS data from a CSV file."""
    df_bts = pd.read_csv(data_path)
    return sorted(list(set(df_bts["ZTFID"])))

def divide_chunks(lst, n):
    """Divide a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_queries(obj_id):
    """Create queries for a given object ID."""
    query_alerts = {
        "query_type": "find",
        "query": {
            "catalog": 'ZTF_alerts',
            "filter": {"objectId": obj_id},
            "projection": {
                "_id": 0,
                "objectId": 1,
                "candidate": 1,
                "cutoutScience": 1,
                "cutoutTemplate": 1,
                "cutoutDifference": 1,
            },
        },
    }
    query_alerts_aux = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts_aux",
            "filter": {"_id": obj_id},
            "projection": {"prv_candidates": 1},
        },
    }
    return query_alerts, query_alerts_aux

def save_alerts(obj_id, data, base_path):
    """Save alerts data to a file."""
    objDirectory = os.path.join(base_path, obj_id)
    if not os.path.exists(objDirectory):
        os.makedirs(objDirectory)
    alertsFile = os.path.join(objDirectory, 'alerts.npy')
    np.save(alertsFile, data)

def save_aux_alerts(obj_id, data, base_path):
    """Save auxiliary alerts data to a CSV file."""
    objDirectory = os.path.join(base_path, obj_id)
    if not os.path.exists(objDirectory):
        os.makedirs(objDirectory)
    photo = data['prv_candidates']
    photo_df = pd.DataFrame(photo)
    photo_df['obj_id'] = obj_id
    photo_df.to_csv(os.path.join(objDirectory, 'photometry.csv'), index=False)

def process_responses(responses, base_path):
    """Process responses from Kowalski and save data."""
    for response in responses["kowalski"]:
        if response["status"] == "success":
            data = response["data"]
            if not data:
                continue
            if "objectId" in data[0]:
                obj_id = data[0]["objectId"]
                save_alerts(obj_id, data, base_path)
            elif "prv_candidates" in data[0]:
                obj_id = data[0]["_id"]
                save_aux_alerts(obj_id, data[0], base_path)

def is_object_processed(obj_id, base_path):
    objDirectory = os.path.join(base_path, obj_id)
    alertsFile = os.path.join(objDirectory, 'alerts.npy')
    photometryFile = os.path.join(objDirectory, 'photometry.csv')
    return os.path.exists(alertsFile) and os.path.exists(photometryFile)

def main():
    ensure_directory_exists(BASE_PATH)
    k = get_kowalski_instance()
    check_kowalski_connection(k)

    objIds = load_bts_data(DATA_PATH)
    num_threads = multiprocessing.cpu_count() - 5
    print(f"Number of threads: {num_threads}")
    total_objects = len(objIds)
    print(f"Total number of objects: {total_objects}")

    batches = list(divide_chunks(objIds, BATCH_SIZE))

    for batch in tqdm(batches, desc="Processing batches"):
        queries = []

        for obj_id in batch:
            if is_object_processed(obj_id, BASE_PATH):
                continue
            query_alerts, query_alerts_aux = create_queries(obj_id)
            queries.append(query_alerts)
            queries.append(query_alerts_aux)

        responses = k.query(queries=queries, use_batch_query=True, max_n_threads=num_threads)
        process_responses(responses, BASE_PATH)