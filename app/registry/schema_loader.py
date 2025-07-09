# app/registry/schema_loader.py
import json
import os
from typing import Dict, Any
import logging
from app.engine.validator import validate_schema
from app.engine.registry import registry
from .version_manager import VersionManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaLoader:
    def __init__(self, schema_dir: str = "schemas"):
        self.schema_dir = schema_dir
        self.version_manager = VersionManager()

    def load_from_file(self, filename: str) -> bool:
        """Load a schema from a JSON file and register it."""
        file_path = os.path.join(self.schema_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"Schema file {file_path} not found")
            return False

        try:
            with open(file_path, 'r') as f:
                schema = json.load(f)
            validate_schema(schema)
            version = self.version_manager.get_version(schema)
            registry.register(schema["pipeline_id"], schema, version)
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error loading schema from {file_path}: {str(e)}")
            return False

    def load_all(self) -> int:
        """Load all schemas from the schema directory."""
        if not os.path.exists(self.schema_dir):
            logger.error(f"Schema directory {self.schema_dir} does not exist")
            return 0
        loaded_count = 0
        for filename in os.listdir(self.schema_dir):
            if filename.endswith('.json'):
                if self.load_from_file(filename):
                    loaded_count += 1
        return loaded_count

def load_schema(client, db_name: str, pipeline_id: str, version: str = None) -> Dict[str, Any]:
    """Load a schema from MongoDB based on pipeline ID and optional version."""
    vm = VersionManager()
    vm.client = client
    vm.db = client[db_name]
    versions = vm.list_versions(pipeline_id)
    if not versions:
        logger.error(f"No versions found for pipeline {pipeline_id}")
        return None
    if version:
        target_version = next((v for v in versions if v["version"] == version), None)
        if not target_version:
            logger.error(f"Version {version} not found for pipeline {pipeline_id}")
            return None
        return target_version["schema"]
    return versions[0]["schema"]  # Return latest version by default

# Example usage
if __name__ == "__main__":
    loader = SchemaLoader()
    loader.load_all()