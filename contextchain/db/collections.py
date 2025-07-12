# app/db/collections.py
from pymongo import MongoClient
from typing import Dict

def setup_collections(client: MongoClient, db_name: str) -> None:
    """
    Set up MongoDB collections for the given database.
    """
    db = client[db_name]

    # Ensure required collections exist
    collections = [
        "schema_registry",  # Stores pipeline schemas and versions
        "task_results",     # Stores task execution results
        "trigger_logs"      # Stores pipeline execution logs
    ]

    for collection in collections:
        db[collection].create_index([("pipeline_id", 1), ("task_id", 1)], unique=True)
        db[collection].create_index("executed_at")

    # Optional: Initialize with a sample document if empty (for testing)
    if db["schema_registry"].count_documents({}) == 0:
        db["schema_registry"].insert_one({"_id": "init", "message": "Database initialized"})