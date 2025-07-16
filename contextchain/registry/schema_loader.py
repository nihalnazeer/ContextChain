#!/usr/bin/env python3
from typing import Dict, Any, List
import logging
import os
import json
from contextchain.engine.validator import validate_schema
from contextchain.db.mongo_client import get_mongo_client
from contextchain.registry.version_manager import VersionManager, push_schema, list_versions, rollback_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaLoader:
    def __init__(self, schema_dir: str = "schemas"):
        self.schema_dir = schema_dir
        self.version_manager = VersionManager()

    def load_from_file(self, filename: str, is_initial: bool = False) -> bool:
        file_path = os.path.join(self.schema_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"Schema file {file_path} not found")
            return False
        try:
            with open(file_path, 'r') as f:
                schema = json.load(f)
            validate_schema(schema, is_initial=is_initial)
            client = get_mongo_client()
            db_name = schema.get("pipeline_id", "contextchain_db")  # Use pipeline_id as db_name
            push_schema(client, db_name, schema, increment=False, is_initial=is_initial)
            logger.info(f"Loaded and registered schema from {file_path} with version {schema['schema_version']}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error loading schema from {file_path}: {str(e)}")
            return False

    def load_all(self) -> int:
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
    vm = VersionManager(client, db_name)
    versions = vm.list_versions(pipeline_id)
    logger.debug(f"Versions retrieved for {pipeline_id}: {versions}")
    if not versions:
        logger.error(f"No versions found for pipeline {pipeline_id} in MongoDB")
        return None
    if version:
        target_version = next((v for v in versions if v["schema_version"] == version), None)
        if not target_version:
            logger.error(f"Version {version} not found for pipeline {pipeline_id}")
            return None
        logger.info(f"Loaded schema version {version} for pipeline {pipeline_id}")
        return target_version["schema"]
    latest = max(versions, key=lambda x: [int(p) for p in x["schema_version"].replace("v", "").split(".")])
    logger.info(f"Loaded latest schema version {latest['schema_version']} for pipeline {pipeline_id}")
    return latest["schema"]