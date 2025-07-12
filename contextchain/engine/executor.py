import logging
from typing import Dict, List, Any
import requests
from contextchain.db.mongo_client import get_mongo_client
from contextchain.engine.validator import validate_schema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_http(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an HTTP task using a real API call."""
    try:
        # Extract allowed domains from schema (assumed to be passed via context or global config)
        schema = task.get("schema", {})  # Assuming schema is passed or accessible
        allowed_domains = schema.get("global_config", {}).get("allowed_domains", [])
        
        # Parse endpoint to check domain
        from urllib.parse import urlparse
        endpoint_domain = urlparse(task["endpoint"]).netloc
        if allowed_domains and endpoint_domain not in allowed_domains:
            raise ValueError(f"Endpoint domain {endpoint_domain} not allowed. Allowed domains: {allowed_domains}")

        # Make the HTTP request
        response = requests.get(task["endpoint"], timeout=10)  # 10-second timeout
        response.raise_for_status()  # Raise exception for 4xx/5xx errors
        
        return {
            "status": "success",
            "output": response.json(),
            "task_id": task["task_id"]
        }
    except requests.RequestException as e:
        logger.error(f"HTTP request failed for task {task['task_id']}: {str(e)}")
        return {"status": "failure", "output": str(e), "task_id": task["task_id"]}
    except ValueError as e:
        logger.error(f"Validation error for task {task['task_id']}: {str(e)}")
        return {"status": "failure", "output": str(e), "task_id": task["task_id"]}

def execute_local(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a local task (simulated)."""
    return {"status": "success", "output": f"processed from {task['endpoint']}", "task_id": task["task_id"]}

def execute_llm(task: Dict[str, Any], context: Dict[int, Any]) -> Dict[str, Any]:
    """Execute an LLM task (simulated)."""
    prompt = task.get("prompt_template", "").format(**context)
    return {"status": "success", "output": f"LLM summary for {prompt}", "task_id": task["task_id"]}

def resolve_dependencies(tasks: List[Dict[str, Any]], task_id: int, context: Dict[int, Any]) -> Dict[str, Any]:
    """
    Resolve dependencies for a given task by collecting outputs from input tasks.
    """
    task = next(t for t in tasks if t["task_id"] == task_id)
    inputs = task.get("inputs", [])
    resolved_context = context.copy()

    for input_id in inputs:
        if input_id not in context:
            logger.error(f"Dependency {input_id} not executed before task {task_id}")
            raise ValueError(f"Dependency {input_id} not executed before task {task_id}")
        resolved_context[input_id] = context[input_id]

    return resolved_context

def execute_single_task(client, db_name: str, schema: Dict[str, Any], task: Dict[str, Any], context: Dict[int, Any] = None) -> Dict[str, Any]:
    """
    Execute a single task based on its task_type.
    """
    if context is None:
        context = resolve_dependencies(schema["tasks"], task["task_id"], {})

    wait_for_input = task.get("wait_for_input", False)
    if wait_for_input and not context:
        raise ValueError("wait_for_input is True but no context available")

    task_id = task["task_id"]
    max_retries = schema["global_config"]["max_retries"]
    retry_on_failure = schema["global_config"]["retry_on_failure"]

    for attempt in range(max_retries + 1):
        try:
            # Pass schema to execute_http for domain validation
            task_with_schema = task.copy()
            task_with_schema["schema"] = schema

            if task["task_type"] == "GET":
                result = execute_http(task_with_schema)
            elif task["task_type"] == "LOCAL":
                result = execute_local(task)
            elif task["task_type"] == "LLM":
                result = execute_llm(task, context)
            else:
                raise ValueError(f"Unsupported task_type: {task['task_type']}")

            # Store result in MongoDB
            db = client[db_name]
            db["task_results"].insert_one(result)
            return result

        except Exception as e:
            logger.error(f"Task {task_id} attempt {attempt + 1} failed: {str(e)}")
            if not retry_on_failure or attempt == max_retries:
                raise
            continue

def execute_pipeline(client, db_name: str, schema: Dict[str, Any]) -> None:
    """
    Execute the entire pipeline of tasks.
    """
    logger.info(f"Starting pipeline execution for {schema['pipeline_id']}")
    validate_schema(schema)

    # Sort tasks by task_id to ensure sequential execution
    tasks = sorted(schema["tasks"], key=lambda x: x["task_id"])
    context = {}

    for task in tasks:
        context = resolve_dependencies(schema["tasks"], task["task_id"], context)
        result = execute_single_task(client, db_name, schema, task, context)
        context[task["task_id"]] = result  # Update context with the current task's result

    logger.info(f"Pipeline {schema['pipeline_id']} execution completed")

if __name__ == "__main__":
    # Example usage
    client = get_mongo_client("mongodb://localhost:27017")
    sample_schema = {
        "pipeline_id": "test_pipeline",
        "schema_version": "v1.0.0",
        "description": "Test pipeline",
        "created_by": "user",
        "created_at": "2025-07-10T01:31:00Z",  # Updated to current time
        "tasks": [
            {"task_id": 1, "description": "Fetch data", "task_type": "GET", "endpoint": "https://jsonplaceholder.typicode.com/posts/1", "inputs": [], "wait_for_input": False},
            {"task_id": 2, "description": "Process data", "task_type": "LOCAL", "endpoint": "path.to.function", "inputs": [1], "wait_for_input": False}
        ],
        "global_config": {"default_output_db": "test_db", "logging_level": "INFO", "retry_on_failure": True, "max_retries": 2, "allowed_task_types": ["GET", "LOCAL"], "allowed_domains": ["jsonplaceholder.typicode.com"]},
        "metadata": {"tags": ["test", "data"], "pipeline_type": "fullstack-ai", "linked_pipelines": []}
    }
    try:
        execute_pipeline(client, "test_db", sample_schema)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")