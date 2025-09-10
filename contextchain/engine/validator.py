#!/usr/bin/env python3
import json
from typing import Dict, List, Any
from urllib.parse import urlparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_schema(schema: Dict[str, Any], is_initial: bool = False) -> None:
    """
    Validate the entire schema structure and its tasks.
    Raises ValueError with detailed error messages if validation fails.

    Args:
        schema (Dict[str, Any]): The schema to validate.
        is_initial (bool): Flag indicating if this is the initial schema creation (no duplicate check).
    """
    logger.debug(f"Validating schema: {schema}")
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")

    required_fields = ["pipeline_id", "schema_version", "description", "created_by", "created_at", "tasks", "global_config", "metadata"]
    missing_fields = [field for field in required_fields if field not in schema]
    if missing_fields:
        raise ValueError(f"Missing required schema fields: {', '.join(missing_fields)}")

    if not isinstance(schema["pipeline_id"], str) or not schema["pipeline_id"].strip():
        raise ValueError("pipeline_id must be a non-empty string")

    if not isinstance(schema["schema_version"], str) or not schema["schema_version"].startswith("v"):
        raise ValueError("schema_version must be a string starting with 'v' (e.g., v1.0.0)")

    if not isinstance(schema["created_at"], str) or not schema["created_at"].endswith("Z"):
        raise ValueError("created_at must be an ISO format string ending with 'Z'")

    if not isinstance(schema["tasks"], list) or not schema["tasks"]:
        raise ValueError("tasks must be a non-empty list")
    validate_tasks(schema["tasks"], schema)

    validate_global_config(schema["global_config"])
    validate_metadata(schema["metadata"])

def validate_tasks(tasks: List[Dict[str, Any]], schema: Dict[str, Any]) -> None:
    """
    Validate the list of tasks and their dependencies.
    """
    task_ids = set()
    allowed_task_types = [
        "GET", "POST", "PUT", "LLM", "LOCAL", "HTTP",
        "VECTOR_STORE_ADD", "VECTOR_STORE_SEARCH", "CHUNK_TEXT",
        "SUMMARIZE", "EVALUATE", "LLM_GENERATE"
    ]
    for task in tasks:
        logger.debug(f"Validating task: {task}")
        if not isinstance(task, dict):
            raise ValueError("Each task must be a dictionary")

        required_task_fields = ["task_id", "description", "task_type"]
        missing_fields = [field for field in required_task_fields if field not in task]
        if missing_fields:
            raise ValueError(f"Task {task.get('task_id', 'unknown')} is missing required fields: {', '.join(missing_fields)}")

        if not isinstance(task["task_id"], int) or task["task_id"] <= 0:
            raise ValueError(f"Task {task['task_id']} must have a positive integer task_id")
        if task["task_id"] in task_ids:
            raise ValueError(f"Duplicate task_id {task['task_id']} found")
        task_ids.add(task["task_id"])

        if not isinstance(task["description"], str) or not task["description"].strip():
            raise ValueError(f"Task {task['task_id']} description must be a non-empty string")

        if task["task_type"] not in allowed_task_types:
            raise ValueError(f"Task {task['task_id']} task_type must be one of {', '.join(allowed_task_types)}")

        # Task-specific validations
        if task["task_type"] in ["GET", "POST", "PUT", "HTTP", "LLM"]:
            if "endpoint" not in task or not isinstance(task["endpoint"], str) or not task["endpoint"].strip():
                raise ValueError(f"Task {task['task_id']} ({task['task_type']}) endpoint must be a non-empty string")

        if task["task_type"] in ["VECTOR_STORE_ADD", "VECTOR_STORE_SEARCH"]:
            if "parameters" not in task or not isinstance(task["parameters"], dict):
                raise ValueError(f"Task {task['task_id']} ({task['task_type']}) requires a parameters dictionary")
            if "collection_name" not in task["parameters"] or not isinstance(task["parameters"]["collection_name"], str):
                raise ValueError(f"Task {task['task_id']} ({task['task_type']}) requires a non-empty string collection_name in parameters")
            if task["task_type"] == "VECTOR_STORE_ADD":
                if "documents" not in task["parameters"] or not isinstance(task["parameters"]["documents"], list):
                    raise ValueError(f"Task {task['task_id']} (VECTOR_STORE_ADD) requires a list of documents in parameters")
            elif task["task_type"] == "VECTOR_STORE_SEARCH":
                if "query" not in task["parameters"] or not isinstance(task["parameters"]["query"], str):
                    raise ValueError(f"Task {task['task_id']} (VECTOR_STORE_SEARCH) requires a string query in parameters")

        if task["task_type"] in ["CHUNK_TEXT", "SUMMARIZE"]:
            if "parameters" not in task or not isinstance(task["parameters"], dict):
                raise ValueError(f"Task {task['task_id']} ({task['task_type']}) requires a parameters dictionary")
            if "text" not in task["parameters"] or not isinstance(task["parameters"]["text"], str):
                raise ValueError(f"Task {task['task_id']} ({task['task_type']}) requires a string text in parameters")

        if task["task_type"] == "LLM_GENERATE":
            if "parameters" not in task or not isinstance(task["parameters"], dict):
                raise ValueError(f"Task {task['task_id']} (LLM_GENERATE) requires a parameters dictionary")
            if "prompt" not in task["parameters"] or not isinstance(task["parameters"]["prompt"], str):
                raise ValueError(f"Task {task['task_id']} (LLM_GENERATE) requires a string prompt in parameters")

        if "inputs" in task and task["inputs"]:
            if not isinstance(task["inputs"], list):
                raise ValueError(f"Task {task['task_id']} inputs must be a list")
            for input_ref in task["inputs"]:
                if not (isinstance(input_ref, int) or isinstance(input_ref, str)):
                    raise ValueError(f"Task {task['task_id']} input {input_ref} must be an integer or string")
                if isinstance(input_ref, int):
                    if input_ref not in task_ids or input_ref >= task["task_id"]:
                        raise ValueError(f"Task {task['task_id']} references non-existent or future input task {input_ref}")

        if "input_mapping" in task and task["input_mapping"]:
            if not isinstance(task["input_mapping"], list):
                raise ValueError(f"Task {task['task_id']} input_mapping must be a list")
            for mapping in task["input_mapping"]:
                if not isinstance(mapping, dict):
                    raise ValueError(f"Task {task['task_id']} input_mapping item must be a dictionary")
                required_fields = ["source", "key", "task_id"]
                missing_fields = [field for field in required_fields if field not in mapping]
                if missing_fields:
                    raise ValueError(f"Task {task['task_id']} input_mapping item missing required fields: {', '.join(missing_fields)}")
                if not isinstance(mapping["task_id"], int):
                    raise ValueError(f"Task {task['task_id']} input_mapping task_id must be an integer")
                if mapping["task_id"] not in task_ids or mapping["task_id"] >= task["task_id"]:
                    raise ValueError(f"Task {task['task_id']} references non-existent or future input task {mapping['task_id']}")
                if not isinstance(mapping["source"], str) or not mapping["source"].strip():
                    raise ValueError(f"Task {task['task_id']} input_mapping source must be a non-empty string")
                if not isinstance(mapping["key"], str) or not mapping["key"].strip():
                    raise ValueError(f"Task {task['task_id']} input_mapping key must be a non-empty string")

        if "target_host" in task:
            if not isinstance(task["target_host"], str) or not task["target_host"].strip():
                raise ValueError(f"Task {task['task_id']} target_host must be a non-empty string")
            backend_hosts = schema["global_config"].get("backend_hosts", {})
            if backend_hosts and task["target_host"] not in backend_hosts:
                raise ValueError(f"Task {task['task_id']} target_host '{task['target_host']}' not found in backend_hosts")

        if "input_source" in task and task["input_source"] is not None:
            if not (isinstance(task["input_source"], str) or isinstance(task["input_source"], list)):
                raise ValueError(f"Task {task['task_id']} input_source must be a string or list")
            if isinstance(task["input_source"], list):
                if not task["input_source"] or len(task["input_source"]) != len(task.get("inputs", [])):
                    raise ValueError(f"Task {task['task_id']} input_source list must match inputs length")
                for source in task["input_source"]:
                    if not isinstance(source, str) or not source.strip():
                        raise ValueError(f"Task {task['task_id']} input_source list items must be non-empty strings")
            else:
                if not task["input_source"].strip():
                    raise ValueError(f"Task {task['task_id']} input_source must be a non-empty string")

        if "wait_for_input" in task:
            if not isinstance(task["wait_for_input"], bool):
                raise ValueError(f"Task {task['task_id']} wait_for_input must be a boolean")

        if "output_collection" in task:
            if not isinstance(task["output_collection"], str) or not task["output_collection"].strip():
                raise ValueError(f"Task {task['task_id']} output_collection must be a non-empty string")

        if "parameters" in task and task["parameters"]:
            if not isinstance(task["parameters"], dict):
                raise ValueError(f"Task {task['task_id']} parameters must be a dictionary")
            for key, value in task["parameters"].items():
                if key in ["max_wait_seconds", "timeout", "max_tokens", "k"]:
                    if not isinstance(value, int):
                        raise ValueError(f"Task {task['task_id']} parameter {key} must be an integer")
                    if value <= 0:
                        raise ValueError(f"Task {task['task_id']} parameter {key} must be positive")
                if task["task_type"] == "LOCAL" and key == "granularity":
                    if not isinstance(value, (int, str)):
                        raise ValueError(f"Task {task['task_id']} parameter {key} must be an integer or string")
                elif key not in ["max_wait_seconds", "timeout", "granularity", "collection_name", "documents", "metadata", "query", "text", "prompt", "max_tokens", "k"]:
                    if not isinstance(value, (int, str, bool, list, dict)):
                        raise ValueError(f"Task {task['task_id']} parameter {key} must be an integer, string, boolean, list, or dict")

        if "rerun" in task:
            if not isinstance(task["rerun"], (type(None), bool)):
                raise ValueError(f"Task {task['task_id']} rerun must be a boolean or null")

        if task["task_type"] == "LLM" and "endpoint" not in task and ("prompt_template" not in task or not task["prompt_template"]):
            raise ValueError(f"Task {task['task_id']} (LLM) requires a prompt_template when not using an endpoint")

        if task["task_type"] == "LLM" and "model" in task:
            if not isinstance(task["model"], str) or not task["model"].strip():
                raise ValueError(f"Task {task['task_id']} model must be a non-empty string")

        if "cron" in task and task["cron"]:
            if not isinstance(task["cron"], str) or not task["cron"].strip():
                raise ValueError(f"Task {task['task_id']} cron must be a non-empty string")

def validate_global_config(config: Dict[str, Any]) -> None:
    """
    Validate the global configuration.
    """
    required_fields = ["default_output_db", "logging_level", "retry_on_failure", "max_retries", "allowed_task_types", "allowed_domains", "backend_host"]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required global_config fields: {', '.join(missing_fields)}")

    if not isinstance(config["default_output_db"], str) or not config["default_output_db"].strip():
        raise ValueError("default_output_db must be a non-empty string")

    allowed_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
    if config["logging_level"] not in allowed_levels:
        raise ValueError(f"logging_level must be one of {', '.join(allowed_levels)}")

    if not isinstance(config["retry_on_failure"], bool):
        raise ValueError("retry_on_failure must be a boolean")

    if not isinstance(config["max_retries"], int) or config["max_retries"] < 0:
        raise ValueError("max_retries must be a non-negative integer")

    if not isinstance(config["allowed_task_types"], list):
        raise ValueError("allowed_task_types must be a list")
    allowed_task_types = [
        "GET", "POST", "PUT", "LLM", "LOCAL", "HTTP",
        "VECTOR_STORE_ADD", "VECTOR_STORE_SEARCH", "CHUNK_TEXT",
        "SUMMARIZE", "EVALUATE", "LLM_GENERATE"
    ]
    for task_type in config["allowed_task_types"]:
        if task_type not in allowed_task_types:
            raise ValueError(f"allowed_task_types contains invalid task type {task_type}")

    if not isinstance(config["allowed_domains"], list):
        raise ValueError("allowed_domains must be a list")
    for domain in config["allowed_domains"]:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("allowed_domains must contain non-empty strings")

    if not isinstance(config["backend_host"], str) or not config["backend_host"].strip():
        raise ValueError("backend_host must be a non-empty string")
    try:
        result = urlparse(config["backend_host"])
        if not all([result.scheme, result.netloc]):
            raise ValueError("backend_host must be a valid URL with scheme and netloc (e.g., http://127.0.0.1:8000)")
    except ValueError as e:
        raise ValueError(f"backend_host validation failed: {str(e)}")

    # Validate vector_db_config
    if "vector_db_config" in config:
        if not isinstance(config["vector_db_config"], dict):
            raise ValueError("vector_db_config must be a dictionary")
        if "type" not in config["vector_db_config"] or config["vector_db_config"]["type"] != "chroma":
            raise ValueError("vector_db_config.type must be 'chroma'")
        if "path" in config["vector_db_config"]:
            if not isinstance(config["vector_db_config"]["path"], str) or not config["vector_db_config"]["path"].strip():
                raise ValueError("vector_db_config.path must be a non-empty string")
            if not Path(config["vector_db_config"]["path"]).is_dir():
                logger.warning(f"vector_db_config.path {config['vector_db_config']['path']} does not exist; will be created on initialization")

    # Validate llm_config
    if "llm_config" in config:
        if not isinstance(config["llm_config"], dict):
            raise ValueError("llm_config must be a dictionary")
        allowed_providers = ["ollama", "openrouter"]
        if "provider" in config["llm_config"] and config["llm_config"]["provider"] not in allowed_providers:
            raise ValueError(f"llm_config.provider must be one of {', '.join(allowed_providers)}")
        if config["llm_config"].get("provider") == "ollama":
            if "model" not in config["llm_config"] or not isinstance(config["llm_config"]["model"], str):
                raise ValueError("llm_config.model must be a non-empty string for ollama provider")
        if "url" in config["llm_config"]:
            if not isinstance(config["llm_config"]["url"], str) or not config["llm_config"]["url"].strip():
                raise ValueError("llm_config.url must be a non-empty string")
            try:
                result = urlparse(config["llm_config"]["url"])
                if not all([result.scheme, result.netloc]):
                    raise ValueError("llm_config.url must be a valid URL with scheme and netloc")
            except ValueError as e:
                raise ValueError(f"llm_config.url validation failed: {str(e)}")
        if "api_key" in config["llm_config"] and not isinstance(config["llm_config"]["api_key"], str):
            raise ValueError("llm_config.api_key must be a string if provided")
        if "api_key_env" in config["llm_config"] and not isinstance(config["llm_config"]["api_key_env"], str):
            raise ValueError("llm_config.api_key_env must be a string if provided")
        if "model" in config["llm_config"] and not isinstance(config["llm_config"]["model"], str):
            raise ValueError("llm_config.model must be a string if provided")
        if "referer" in config["llm_config"] and not isinstance(config["llm_config"]["referer"], str):
            raise ValueError("llm_config.referer must be a string if provided")
        if "title" in config["llm_config"] and not isinstance(config["llm_config"]["title"], str):
            raise ValueError("llm_config.title must be a string if provided")

def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validate the metadata section.
    """
    required_fields = ["tags", "pipeline_type", "linked_pipelines"]
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        raise ValueError(f"Missing required metadata fields: {', '.join(missing_fields)}")

    if not isinstance(metadata["tags"], list):
        raise ValueError("tags must be a list")
    for tag in metadata["tags"]:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("tags must contain non-empty strings")

    if not isinstance(metadata["pipeline_type"], str) or not metadata["pipeline_type"].strip():
        raise ValueError("pipeline_type must be a non-empty string")

    if not isinstance(metadata["linked_pipelines"], list):
        raise ValueError("linked_pipelines must be a list")
    for pipeline in metadata["linked_pipelines"]:
        if not isinstance(pipeline, str) or not pipeline.strip():
            raise ValueError("linked_pipelines must contain non-empty strings")

if __name__ == "__main__":
    with open("/Users/mohammednihal/Desktop/Business Intelligence/AgentBI/Backend/schemas/AgentBI-Demo.json", "r") as f:
        sample_schema = json.load(f)
    try:
        validate_schema(sample_schema)
        print("Schema is valid!")
    except ValueError as e:
        print(f"Validation failed: {str(e)}")