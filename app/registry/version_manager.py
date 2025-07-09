from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionManager:
    def __init__(self):
        self.supported_versions = {"v1.0.0"}  # Extend this as needed

    def get_version(self, schema: Dict[str, Any]) -> str:
        """
        Returns the supported version from the schema.
        If unsupported or missing, logs a warning and returns default.
        """
        version = schema.get("schema_version", "v1.0.0")
        if version not in self.supported_versions:
            logger.warning(
                f"Version {version} not in supported versions {self.supported_versions}. Using default v1.0.0"
            )
            return "v1.0.0"
        return version

    def get_raw_version(self, schema: Dict[str, Any]) -> str:
        """
        Get the schema version without fallback logic.
        """
        return schema.get("schema_version")

    def is_compatible(self, schema: Dict[str, Any]) -> bool:
        """
        Strictly checks whether the schema version is in supported versions.
        """
        raw_version = self.get_raw_version(schema)
        return raw_version in self.supported_versions

    def update_version(self, schema: Dict[str, Any], new_version: str) -> Dict[str, Any]:
        """
        Update the version of a schema to a new supported version.
        """
        if new_version not in self.supported_versions:
            logger.error(f"Version {new_version} not supported")
            raise ValueError(f"Version {new_version} not supported")
        schema["schema_version"] = new_version
        logger.info(f"Updated schema version to {new_version}")
        return schema

# Example usage
if __name__ == "__main__":
    vm = VersionManager()
    sample_schema = {"schema_version": "v1.0.0", "pipeline_id": "test"}
    print(vm.get_version(sample_schema))        # v1.0.0
    print(vm.is_compatible(sample_schema))      # True
