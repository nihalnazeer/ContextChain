import logging
import requests
import json
import os
from typing import Dict, List, Any
from contextchain.db.mongo_client import get_mongo_client
from contextchain.engine.validator import validate_schema
from urllib.parse import urljoin, urlparse
from datetime import datetime
import time
import importlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resolve_dependencies(tasks: List[Dict[str, Any]], task_id: int, context: Dict[int, Any]) -> Dict[str, Any]:
    """Resolve dependencies and build input mapping."""
    task = next(t for t in tasks if t["task_id"] == task_id)
    inputs = task.get("inputs", [])
    input_mapping = task.get("input_mapping", [])
    resolved_context = context.copy()

    for input_id in inputs:
        if input_id not in context:
            raise ValueError(f"Dependency {input_id} not executed before task {task_id}")
        resolved_context[input_id] = context[input_id]

    payload = {}
    for mapping in input_mapping:
        source = mapping.get("source", "task_results")
        key = mapping.get("key")
        task_id_ref = mapping.get("task_id")
        if task_id_ref and task_id_ref in context:
            data = context[task_id_ref]
            if isinstance(data, dict) and key in data.get("output", {}):
                payload[key] = data["output"][key]
    return payload

def execute_http_request(url: str, method: str, payload: Dict[str, Any], headers: Dict[str, str] = None, timeout: int = 30, retries: int = 0) -> Dict[str, Any]:
    """Execute an HTTP request with retry logic."""
    headers = headers or {"Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            response = requests.request(method.lower(), url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return {"output": response.json(), "status": "success"}
        except requests.RequestException as e:
            logger.error(f"HTTP request failed (attempt {attempt + 1}/{retries + 1}): {str(e)}")
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return {"status": "failed"}

def execute_llm_request(llm_config: Dict[str, Any], prompt: str, task_model: str = None, timeout: int = 30) -> Dict[str, Any]:
    """Execute an LLM request using global config."""
    # Try direct API key first, fall back to environment variable
    api_key = llm_config.get("api_key")
    if not api_key:
        api_key_env = llm_config.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError("LLM API key not found in schema or environment variables")
    
    if llm_config.get("api_key"):
        logger.info("Using API key directly from schema (not recommended for security).")
    else:
        logger.info(f"Using API key from environment variable: {api_key_env}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": llm_config.get("referer", "http://your-site-url.com"),
        "X-Title": llm_config.get("title", "Your Site Name"),
        "Content-Type": "application/json"
    }
    model = task_model or llm_config.get("model", "mistralai/mistral-small-3.2-24b-instruct:free")
    response = requests.post(
        llm_config["url"],
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=timeout
    )
    response.raise_for_status()
    return {"output": response.json()["choices"][0]["message"]["content"], "status": "success"}

def execute_task(client, db_name: str, schema: Dict[str, Any], task: Dict[str, Any], context: Dict[int, Any] = None) -> Dict[str, Any]:
    """Execute a single task based on task_type."""
    if context is None:
        context = {}

    task_id = task["task_id"]
    global_config = schema["global_config"]
    max_retries = global_config.get("max_retries", 2)
    retry_on_failure = global_config.get("retry_on_failure", True)

    payload = resolve_dependencies(schema["tasks"], task_id, context)
    payload.update(task.get("parameters", {}))

    result = {"task_id": task_id}
    try:
        if task["task_type"] == "LOCAL":
            module_path, func_name = task["endpoint"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            output = func(**payload)
        elif task["task_type"] in ["HTTP", "POST", "GET", "PUT"]:
            endpoint = task["endpoint"]
            if not urlparse(endpoint).scheme:
                backend_host = global_config.get("backend_hosts", {}).get(task.get("target_host", "default"), "http://127.0.0.1:8080")
                url = urljoin(backend_host, endpoint.lstrip("/"))
            else:
                url = endpoint
            headers = task.get("headers", {}) or global_config.get("default_headers", {"Content-Type": "application/json"})
            method = task["task_type"].lower() if task["task_type"] != "HTTP" else "post"
            output = execute_http_request(url, method, payload, headers, timeout=30, retries=max_retries)
        elif task["task_type"] == "LLM":
            llm_config = global_config.get("llm_config", {})
            if not llm_config.get("url"):
                raise ValueError("LLM task requires llm_config.url in global_config")
            prompt = task.get("prompt_template", "").format(**payload)
            task_model = task.get("model")  # Allow task-specific model override
            output = execute_llm_request(llm_config, prompt, task_model)
        else:
            raise ValueError(f"Unsupported task_type: {task['task_type']}")

        result["output"] = output.get("output", output)
        result["status"] = output.get("status", "success")
        db = client[db_name]
        db[task["output_collection"]].insert_one(result)
        context[task_id] = result
        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Task {task_id} execution failed: {str(e)}")
        if not retry_on_failure:
            raise
        return result

def execute_pipeline(client, db_name: str, schema: Dict[str, Any]) -> None:
    """Execute the entire pipeline and generate resolved schema."""
    logger.info(f"Starting pipeline execution for {schema['pipeline_id']}")
    validate_schema(schema)

    tasks = sorted(schema["tasks"], key=lambda x: x["task_id"])
    context = {}

    resolved_schema = schema.copy()
    for task in tasks:
        task_copy = task.copy()
        if task["task_type"] in ["HTTP", "POST", "GET", "PUT"]:
            endpoint = task["endpoint"]
            if not urlparse(endpoint).scheme:
                backend_host = schema["global_config"]["backend_hosts"].get(task.get("target_host", "default"), schema["global_config"].get("backend_host", "http://127.0.0.1:8080"))
                task_copy["full_url"] = urljoin(backend_host, endpoint.lstrip("/"))
            else:
                task_copy["full_url"] = endpoint
        resolved_schema["tasks"] = [t for t in resolved_schema["tasks"] if t["task_id"] != task["task_id"]]
        resolved_schema["tasks"].append(task_copy)

    output_dir = os.path.join("resolved_schema", f"{schema['pipeline_id']}_{schema['schema_version']}.json")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w") as f:
        json.dump(resolved_schema, f, indent=2)

    for task in tasks:
        context = resolve_dependencies(schema["tasks"], task["task_id"], context)
        execute_task(client, db_name, schema, task, context)

    logger.info(f"Pipeline {schema['pipeline_id']} execution completed")

def execute_single_task(client, db_name: str, schema: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single task via API request."""
    context = {}
    return execute_task(client, db_name, schema, task, context)