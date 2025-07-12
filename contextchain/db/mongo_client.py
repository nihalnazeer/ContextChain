from pymongo import MongoClient
import subprocess
import os
import time
from pathlib import Path

def get_mongo_client(uri="mongodb://localhost:27017", db_name="contextchain_db"):
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        return client
    except Exception as e:
        print(f"✗ Failed to connect to MongoDB at {uri}: {e}")
        print("Attempting to start a local MongoDB instance...")
        data_dir = Path("/tmp/contextchain_mongodb")
        data_dir.mkdir(exist_ok=True)
        global mongod_process
        mongod_process = subprocess.Popen([
            "mongod",
            "--dbpath", str(data_dir),
            "--port", "27017",
            "--bind_ip", "127.0.0.1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()
            print("✓ Local MongoDB instance started successfully.")
            return client
        except Exception as e:
            mongod_process.terminate()
            raise Exception(f"✗ Failed to start local MongoDB: {e}")
    return client