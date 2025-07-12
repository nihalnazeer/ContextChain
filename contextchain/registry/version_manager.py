# app/registry/version_manager.py
from typing import Dict, Any, List
import logging
from contextchain.db.mongo_client import get_mongo_client
import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionManager:
    _instance = None
    supported_versions = {"v1.0.0"}  # Define as class variable

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VersionManager, cls).__new__(cls)
            try:
                cls._instance.client = get_mongo_client()  # Initialize client
                cls._instance.db = cls._instance.client["pipeline_db"]
                cls._instance.version_collection = cls._instance.db["versions"]
            except Exception as e:
                logger.error(f"Failed to initialize MongoClient: {str(e)}")
                cls._instance.client = None  # Fallback to None if connection fails
        return cls._instance

    def get_client(self):
        """Return the MongoClient instance, or raise an error if not initialized."""
        if self._instance.client is None:
            raise RuntimeError("MongoClient not initialized. Check connection settings.")
        return self._instance.client

    def get_version(self, schema: Dict[str, Any]) -> str:
        """Extract the version from the schema with fallback to supported version."""
        version = schema.get("schema_version", "v1.0.0")
        if version not in self.supported_versions:
            logger.warning(f"Version {version} not in supported versions {self.supported_versions}. Using default v1.0.0")
        return version if version in self.supported_versions else "v1.0.0"

    def get_raw_version(self, schema: Dict[str, Any]) -> str:
        """Extract the raw version from the schema without fallback."""
        return schema.get("schema_version", "v1.0.0")

    def is_compatible(self, schema: Dict[str, Any]) -> bool:
        """Check if the raw schema version is compatible."""
        raw_version = self.get_raw_version(schema)
        return raw_version in self.supported_versions

    def update_version(self, schema: Dict[str, Any], new_version: str) -> Dict[str, Any]:
        """Update the schema to a new version."""
        if new_version not in self.supported_versions:
            logger.error(f"Version {new_version} not supported")
            raise ValueError(f"Version {new_version} not supported")
        schema["schema_version"] = new_version
        logger.info(f"Updated schema version to {new_version}")
        return schema

    def push_schema(self, pipeline_id: str, schema: Dict[str, Any]) -> str:
        """Save a schema version to MongoDB."""
        version = self.get_version(schema)
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        version_doc = {
            "pipeline_id": pipeline_id,
            "schema": schema,
            "version": version,
            "timestamp": timestamp
        }
        self.version_collection.insert_one(version_doc)
        logger.info(f"Pushed schema for {pipeline_id} with version {version} at {timestamp}")
        return version

    def list_versions(self, pipeline_id: str) -> List[Dict[str, Any]]:
        """Retrieve all versions for a pipeline."""
        versions = list(self.version_collection.find({"pipeline_id": pipeline_id}).sort("timestamp", -1))
        if not versions:
            logger.warning(f"No versions found for pipeline {pipeline_id}")
        return versions

    def rollback_version(self, pipeline_id: str, timestamp: str) -> Dict[str, Any]:
        """Rollback to a specific version based on timestamp."""
        version = self.version_collection.find_one({"pipeline_id": pipeline_id, "timestamp": timestamp})
        if not version:
            logger.error(f"No version found for {pipeline_id} at {timestamp}")
            raise KeyError(f"Version not found for {pipeline_id} at {timestamp}")
        schema = version["schema"]
        logger.info(f"Rolled back {pipeline_id} to version at {timestamp}")
        return schema

    def close(self):
        """Explicitly close the MongoDB client."""
        if hasattr(self._instance, 'client') and self._instance.client:
            self._instance.client.close()
            logger.info("MongoDB client closed")
            self._instance.client = None

# Module-level functions for CLI compatibility
def push_schema(client, db_name: str, schema: Dict[str, Any]) -> str:
    """Push a schema to MongoDB with versioning."""
    vm = VersionManager()
    vm.client = client if client else vm.client
    vm.db = client[db_name] if client else vm.db
    return vm.push_schema(schema["pipeline_id"], schema)

def list_versions(client, db_name: str, pipeline_id: str) -> List[Dict[str, Any]]:
    """List all versions for a pipeline."""
    vm = VersionManager()
    vm.client = client if client else vm.client
    vm.db = client[db_name] if client else vm.db
    return vm.list_versions(pipeline_id)

def rollback_version(client, db_name: str, pipeline_id: str, timestamp: str) -> Dict[str, Any]:
    """Rollback to a previous schema version."""
    vm = VersionManager()
    vm.client = client if client else vm.client
    vm.db = client[db_name] if client else vm.db
    return vm.rollback_version(pipeline_id, timestamp)

# Example usage
if __name__ == "__main__":
    sample_schema = {
        "pipeline_id": "test_pipeline",
        "schema_version": "v1.0.0",
        "description": "Test schema",
        "created_by": "mohammednihal",
        "created_at": "2025-07-10T03:38:00Z",  # Updated to current time
        "tasks": [],
        "global_config": {},
        "metadata": {}
    }
    push_schema(None, None, sample_schema)
    versions = list_versions(None, None, "test_pipeline")
    for v in versions:
        print(v)
    rollback_schema = rollback_version(None, None, "test_pipeline", versions[0]["timestamp"])
    print(rollback_schema)
    VersionManager().close()  # Explicit cleanup