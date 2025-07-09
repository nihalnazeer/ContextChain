# app/db/mongo_client.py
from pymongo import MongoClient
from pathlib import Path
import yaml
import os

def get_mongo_client(uri: str = None) -> MongoClient:
    """
    Get a MongoDB client instance with the specified URI or from config/environment.
    """
    # Prioritize environment variable MONGO_URI
    if uri is None:
        uri = os.getenv("MONGO_URI")
        if uri is None:
            config_path = Path("config/default_config.yaml")
            if config_path.exists():
                with config_path.open("r") as f:
                    config = yaml.safe_load(f)
                uri = config.get("uri", "mongodb://localhost:27017")
            else:
                uri = "mongodb://localhost:27017"
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB at {uri}: {str(e)}")