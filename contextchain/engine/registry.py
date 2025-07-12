# app/engine/registry.py
from typing import Dict, Any, List
import logging
from contextchain.registry.version_manager import VersionManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Registry:
    _instance = None
    _registry: Dict[str, Dict[str, Any]] = {}  # {pipeline_id: {"schema": schema, "version": version}}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Registry, cls).__new__(cls)
        return cls._instance

    def register(self, pipeline_id: str, schema: Dict[str, Any], version: str) -> bool:
        """Register a pipeline schema with a version."""
        if pipeline_id in self._registry:
            logger.warning(f"Overwriting existing pipeline {pipeline_id}")
        self._registry[pipeline_id] = {"schema": schema, "version": version}
        logger.info(f"Registered pipeline {pipeline_id} with version {version}")
        return True

    def get_schema(self, pipeline_id: str) -> Dict[str, Any]:
        """Retrieve the schema for a given pipeline ID."""
        if pipeline_id not in self._registry:
            logger.error(f"Pipeline {pipeline_id} not found in registry")
            raise KeyError(f"Pipeline {pipeline_id} not found")
        return self._registry[pipeline_id]["schema"]

    def get_version(self, pipeline_id: str) -> str:
        """Retrieve the version for a given pipeline ID."""
        if pipeline_id not in self._registry:
            logger.error(f"Pipeline {pipeline_id} not found in registry")
            raise KeyError(f"Pipeline {pipeline_id} not found")
        return self._registry[pipeline_id]["version"]

    def list_pipelines(self) -> List[str]:
        """List all registered pipeline IDs."""
        return list(self._registry.keys())

# Singleton instance
registry = Registry()