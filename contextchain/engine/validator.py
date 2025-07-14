import json
from typing import Dict, List, Any
from urllib.parse import urlparse

def validate_schema(schema: Dict[str, Any]) -> None:
    """
    Validate the entire schema structure and its tasks.
    Raises ValueError with detailed error messages if validation fails.
    """
    print(f"Validating schema: {schema}")  # Debug: Print the entire schema
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
    for task in tasks:
        print(f"Validating task: {task}")  # Debug: Print each task
        if not isinstance(task, dict):
            raise ValueError("Each task must be a dictionary")

        required_task_fields = ["task_id", "description", "task_type", "endpoint"]
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

        allowed_task_types = ["GET", "POST", "PUT", "LLM", "LOCAL", "HTTP"]
        if task["task_type"] not in allowed_task_types:
            raise ValueError(f"Task {task['task_id']} task_type must be one of {', '.join(allowed_task_types)}")

        if not isinstance(task["endpoint"], str) or not task["endpoint"].strip():
            raise ValueError(f"Task {task['task_id']} endpoint must be a non-empty string")

        if "inputs" in task and task["inputs"]:
            if not isinstance(task["inputs"], list):
                raise ValueError(f"Task {task['task_id']} inputs must be a list")
            for input_id in task["inputs"]:
                if not isinstance(input_id, int):
                    raise ValueError(f"Task {task['task_id']} input {input_id} must be an integer")
                if input_id not in task_ids or input_id >= task["task_id"]:
                    raise ValueError(f"Task {task['task_id']} references non-existent or future input task {input_id}")

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

        if "input_source" in task and task["input_source"]:
            if not isinstance(task["input_source"], str) or not task["input_source"].strip():
                raise ValueError(f"Task {task['task_id']} input_source must be a non-empty string")

        if "parameters" in task and task["parameters"]:
            if not isinstance(task["parameters"], dict):
                raise ValueError(f"Task {task['task_id']} parameters must be a dictionary")
            for key, value in task["parameters"].items():
                if key in ["max_wait_seconds", "timeout"] and not isinstance(value, int):
                    raise ValueError(f"Task {task['task_id']} parameter {key} must be an integer")
                if key == "max_wait_seconds" and value <= 0:
                    raise ValueError(f"Task {task['task_id']} max_wait_seconds must be positive")
                if key == "timeout" and value <= 0:
                    raise ValueError(f"Task {task['task_id']} timeout must be positive")

        if task["task_type"] == "LLM" and "endpoint" not in task and ("prompt_template" not in task or not task["prompt_template"]):
            raise ValueError(f"Task {task['task_id']} (LLM) requires a prompt_template when not using an endpoint")

        if task["task_type"] == "LLM" and "model" in task:
            if not isinstance(task["model"], str) or not task["model"].strip():
                raise ValueError(f"Task {task['task_id']} model must be a non-empty string")

        if "output_collection" in task and not isinstance(task["output_collection"], str):
            raise ValueError(f"Task {task['task_id']} output_collection must be a string")

        if "cron" in task and task["cron"]:
            if not isinstance(task["cron"], str) or not task["cron"].strip():
                raise ValueError(f"Task {task['task_id']} cron must be a non-empty string")

def validate_global_config(config: Dict[str, Any]) -> None:
    """
    Validate the global configuration.
    """
    required_fields = ["default_output_db", "logging_level", "retry_on_failure", "max_retries", "allowed_task_types", "allowed_domains"]
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
    allowed_task_types = ["GET", "POST", "PUT", "LLM", "LOCAL", "HTTP"]
    for task_type in config["allowed_task_types"]:
        if task_type not in allowed_task_types:
            raise ValueError(f"allowed_task_types contains invalid task type {task_type}")

    if not isinstance(config["allowed_domains"], list):
        raise ValueError("allowed_domains must be a list")
    for domain in config["allowed_domains"]:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("allowed_domains must contain non-empty strings")

    if "backend_host" in config:
        if not isinstance(config["backend_host"], str) or not config["backend_host"].strip():
            raise ValueError("backend_host must be a non-empty string")
        try:
            result = urlparse(config["backend_host"])
            if not all([result.scheme, result.netloc]):
                raise ValueError("backend_host must be a valid URL with scheme and netloc (e.g., http://127.0.0.1:8000)")
        except ValueError as e:
            raise ValueError(f"backend_host validation failed: {str(e)}")

    if "backend_hosts" in config:
        if not isinstance(config["backend_hosts"], dict):
            raise ValueError("backend_hosts must be a dictionary")
        for host_key, host_value in config["backend_hosts"].items():
            if not isinstance(host_key, str) or not host_key.strip():
                raise ValueError(f"backend_hosts key '{host_key}' must be a non-empty string")
            if not isinstance(host_value, str) or not host_value.strip():
                raise ValueError(f"backend_hosts value for '{host_key}' must be a non-empty string")
            try:
                result = urlparse(host_value)
                if not all([result.scheme, result.netloc]):
                    raise ValueError(f"backend_hosts value for '{host_key}' must be a valid URL with scheme and netloc")
            except ValueError as e:
                raise ValueError(f"backend_hosts value for '{host_key}' validation failed: {str(e)}")

    if "llm_config" in config:
        if not isinstance(config["llm_config"], dict):
            raise ValueError("llm_config must be a dictionary")
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
    sample_schema = {
        "pipeline_id": "test_pipeline",
        "schema_version": "v1.0.0",
        "description": "Test pipeline",
        "created_by": "user",
        "created_at": "2025-07-09T23:39:00Z",
        "tasks": [
            {
                "task_id": 1,
                "description": "Fetch data",
                "task_type": "GET",
                "endpoint": "http://example.com/data",
                "inputs": [],
                "input_mapping": [],
                "target_host": "default"
            },
            {
                "task_id": 2,
                "description": "Generate summary",
                "task_type": "LLM",
                "endpoint": "services.custom_llm.generate_summary",
                "inputs": [1],
                "input_mapping": [{"source": "task_results", "key": "data", "task_id": 1}],
                "parameters": {"output_file": "summary.txt"}
            },
            {
                "task_id": 3,
                "description": "Generate LLM summary",
                "task_type": "LLM",
                "prompt_template": "Summarize {data}: {insights}",
                "inputs": [1],
                "input_mapping": [{"source": "task_results", "key": "data", "task_id": 1}],
                "model": "mistralai/mixtral-8x7b-instruct"
            }
        ],
        "global_config": {
            "default_output_db": "test_db",
            "logging_level": "INFO",
            "retry_on_failure": True,
            "max_retries": 2,
            "allowed_task_types": ["GET", "LLM", "LOCAL", "HTTP"],
            "allowed_domains": ["example.com"],
            "backend_hosts": {
                "default": "http://127.0.0.1:8000",
                "external": "https://example.com"
            },
            "llm_config": {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "api_key": "",  # Direct key
                "api_key_env": "OPENROUTER_API_KEY",  # Optional env variable
                "model": "mistralai/mistral-small-3.2-24b-instruct:free"
            }
        },
        "metadata": {
            "tags": ["test", "data"],
            "pipeline_type": "fullstack-ai",
            "linked_pipelines": []
        }
    }
    try:
        validate_schema(sample_schema)
        print("Schema is valid!")
    except ValueError as e:
        print(f"Validation failed: {str(e)}")