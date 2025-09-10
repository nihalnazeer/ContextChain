#!/usr/bin/env python3
import logging
import requests
import json
import os
from typing import Dict, List, Any
from contextchain.db.mongo_client import get_mongo_client
from contextchain.db.vector_db_client import get_vector_db_client
from contextchain.engine.validator import validate_schema
from contextchain.local_llm_client import OllamaClient
from contextchain.data_processing import chunk_text, summarize_text
from contextchain.dag_builder import build_dag
from contextchain.evaluation import evaluate_results
from contextchain.task_registry import get_task_handler
from datetime import datetime
import time
import importlib
from urllib.parse import urlparse, urljoin
from pymongo import UpdateOne
import concurrent.futures
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_inputs(task: Dict[str, Any], db: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch inputs from previous tasks or input_source based on task dependencies."""
    inputs = {}
    if task.get("inputs"):
        for input_ref in task["inputs"]:
            if isinstance(input_ref, int):  # Task ID dependency
                source_task = next((t for t in schema["tasks"] if t["task_id"] == input_ref), None)
                if source_task:
                    source_coll = source_task.get("output_collection", "task_results")
                    data = db[source_coll].find_one({"task_id": input_ref}, sort=[("timestamp", -1)])
                    inputs[f"task_{input_ref}"] = data.get("output", {}) if data else {}
            elif isinstance(input_ref, str):  # Named input
                source_coll = task.get("input_source")
                if source_coll and isinstance(source_coll, (str, list)):
                    if isinstance(source_coll, list):
                        source_coll = source_coll[0]  # Use first source for now
                    data = db[source_coll].find_one(sort=[("timestamp", -1)])
                    inputs[input_ref] = data.get(input_ref) if data else None
    return inputs

def resolve_dependencies(tasks: List[Dict[str, Any]], task_id: int, context: Dict[int, Any]) -> Dict[str, Any]:
    """Resolve dependencies and build input mapping."""
    task = next(t for t in tasks if t["task_id"] == task_id)
    inputs = task.get("inputs", [])
    input_mapping = task.get("input_mapping", [])
    resolved_context = context.copy()

    for input_id in inputs:
        if isinstance(input_id, int) and input_id not in context:
            raise ValueError(f"Dependency {input_id} not executed before task {task_id}")
        resolved_context[input_id] = context.get(input_id, {})

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
    api_key = llm_config.get("api_key")
    if not api_key or api_key == "":
        api_key_env = llm_config.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError("LLM API key not found in environment variable")

    logger.info(f"Using API key from environment variable: {api_key_env}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": llm_config.get("referer", "http://localhost:3000"),
        "X-Title": llm_config.get("title", "AgentBI-Demo"),
        "Content-Type": "application/json"
    }
    model = task_model or llm_config["model"]
    url = llm_config["url"]
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=timeout
    )
    response.raise_for_status()
    return {"output": response.json()["choices"][0]["message"]["content"], "status": "success"}

def execute_task(task: Dict[str, Any], schema: Dict[str, Any], db: Any) -> Dict[str, Any]:
    """Execute a single task based on its type."""
    output_collection = task.get("output_collection", "task_results")
    result = {}
    parameters = task.get("parameters", {})

    if task["task_type"] == "LLM":
        prompt = task.get("prompt_template", "").format(**parameters)
        result = execute_llm_request(schema["global_config"]["llm_config"], prompt, task.get("model"))
    elif task["task_type"] == "LOCAL":
        module, func = task["endpoint"].rsplit(".", 1)
        try:
            mod = importlib.import_module(module)
            func = getattr(mod, func)
            inputs = fetch_inputs(task, db, schema)
            merged_inputs = {k: v for d in inputs.values() for k, v in d.items()}
            result = func(**merged_inputs, db=db) if merged_inputs else func(db=db)
        except ImportError as e:
            logger.error(f"Task {task['task_id']} execution failed: {str(e)}")
            raise
        except AttributeError as e:
            logger.error(f"Task {task['task_id']} function {func} not found in module {module}: {str(e)}")
            raise
    elif task["task_type"] in ["GET", "POST", "PUT", "HTTP"]:
        full_url = task.get("full_url", task["endpoint"])
        if not urlparse(full_url).scheme:
            backend_host = schema["global_config"]["backend_hosts"].get(task.get("target_host", "default"), schema["global_config"].get("backend_host", "http://127.0.0.1:8000"))
            full_url = urljoin(backend_host, full_url.lstrip("/"))
        result = execute_http_request(full_url, task["task_type"], fetch_inputs(task, db, schema), retries=schema["global_config"]["max_retries"])
    elif task["task_type"] == "VECTOR_STORE_ADD":
        vdb_client = get_vector_db_client(schema["global_config"].get("vector_db_config", {}).get("path", "./contextchain_chromadb"))
        documents = parameters.get("documents", [])
        metadata = parameters.get("metadata", None)
        metrics = vdb_client.add_documents(parameters["collection_name"], documents, metadata)
        result = {"output": metrics, "status": "success"}
        db["metrics"].insert_one({
            "pipeline_id": schema["pipeline_id"],
            "task_id": task["task_id"],
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    elif task["task_type"] == "VECTOR_STORE_SEARCH":
        vdb_client = get_vector_db_client(schema["global_config"].get("vector_db_config", {}).get("path", "./contextchain_chromadb"))
        query = parameters.get("query", "")
        k = parameters.get("k", 5)
        metrics = vdb_client.search(parameters["collection_name"], query, k)
        result = {"output": metrics["results"], "status": "success"}
        db["metrics"].insert_one({
            "pipeline_id": schema["pipeline_id"],
            "task_id": task["task_id"],
            "metrics": {"time_taken": metrics["time_taken"]},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    elif task["task_type"] == "CHUNK_TEXT":
        text = parameters.get("text", "")
        max_length = parameters.get("max_length", 512)
        chunks = chunk_text(text, max_length)
        result = {"output": {"chunks": chunks}, "status": "success"}
    elif task["task_type"] == "SUMMARIZE":
        text = parameters.get("text", "")
        model = schema["global_config"].get("llm_config", {}).get("model", "mistral:7b")
        summary = summarize_text(text, model)
        result = {"output": {"summary": summary}, "status": "success"}
    elif task["task_type"] == "EVALUATE":
        results = list(db["task_results"].find({"task_id": task["task_id"]}))
        evaluation = evaluate_results(results)
        result = {"output": {"evaluation": evaluation}, "status": "success"}
    elif task["task_type"] == "LLM_GENERATE":
        llm_client = OllamaClient(model=schema["global_config"].get("llm_config", {}).get("model", "mistral:7b"))
        prompt = parameters.get("prompt", "")
        max_tokens = parameters.get("max_tokens", 512)
        response = llm_client.generate(prompt, max_tokens)
        result = {"output": {"response": response}, "status": "success"}
    else:
        handler = get_task_handler(task["task_type"])
        if handler:
            inputs = fetch_inputs(task, db, schema)
            result = handler(task, schema, inputs)
        else:
            raise ValueError(f"Unknown task type: {task['task_type']}")

    # Handle rerun logic if enabled
    if task.get("rerun", False):
        existing_result = db[output_collection].find_one({
            "pipeline_id": schema["pipeline_id"],
            "schema_version": schema["schema_version"],
            "task_id": task["task_id"]
        })
        if existing_result:
            previous_data = existing_result.get("output", {})
        else:
            previous_data = {}

        if isinstance(result, dict):
            updated_result = result.copy()
            updated_result.update({
                "pipeline_id": schema["pipeline_id"],
                "schema_version": schema["schema_version"],
                "task_id": task["task_id"],
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "previous_data": previous_data
            })
            db[output_collection].delete_many({
                "pipeline_id": schema["pipeline_id"],
                "schema_version": schema["schema_version"],
                "task_id": task["task_id"]
            })
            db[output_collection].insert_one(updated_result)
            return updated_result
        elif isinstance(result, list):
            updated_results = []
            for item in result:
                if isinstance(item, dict):
                    updated_item = item.copy()
                    updated_item.update({
                        "pipeline_id": schema["pipeline_id"],
                        "schema_version": schema["schema_version"],
                        "task_id": task["task_id"],
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "previous_data": previous_data if not updated_results else {}
                    })
                    db[output_collection].delete_many({
                        "pipeline_id": schema["pipeline_id"],
                        "schema_version": schema["schema_version"],
                        "task_id": task["task_id"],
                        "granularity": updated_item.get("granularity")
                    })
                    db[output_collection].insert_one(updated_item)
                    updated_results.append(updated_item)
                else:
                    logger.warning(f"Skipping non-dictionary item in list for task {task['task_id']}")
            return updated_results
    else:
        # Default behavior: Store as new entry without overwriting
        if isinstance(result, dict):
            updated_result = result.copy()
            updated_result.update({
                "pipeline_id": schema["pipeline_id"],
                "schema_version": schema["schema_version"],
                "task_id": task["task_id"],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            db[output_collection].insert_one(updated_result)
            return updated_result
        elif isinstance(result, list):
            updated_results = []
            for item in result:
                if isinstance(item, dict):
                    updated_item = item.copy()
                    updated_item.update({
                        "pipeline_id": schema["pipeline_id"],
                        "schema_version": schema["schema_version"],
                        "task_id": task["task_id"],
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    db[output_collection].insert_one(updated_item)
                    updated_results.append(updated_item)
                else:
                    logger.warning(f"Skipping non-dictionary item in list for task {task['task_id']}")
            return updated_results

    raise ValueError(f"Unsupported result type {type(result)} for task {task['task_id']}")

def execute_pipeline(client: Any, db_name: str, schema: Dict[str, Any]) -> None:
    """Execute the entire pipeline with parallel task execution."""
    logger.info(f"Starting pipeline execution for {schema['pipeline_id']}")
    db = client[db_name]

    # Validate schema
    validate_schema(schema)

    # Build DAG for parallel execution
    dag = build_dag(schema["tasks"])
    tasks = schema["tasks"]
    context = {}  # Store task results by task_id

    # Resolve URLs for HTTP tasks
    resolved_schema = schema.copy()
    for task in tasks:
        task_copy = task.copy()
        if task["task_type"] in ["HTTP", "POST", "GET", "PUT"]:
            endpoint = task["endpoint"]
            if not urlparse(endpoint).scheme:
                backend_host = schema["global_config"]["backend_hosts"].get(task.get("target_host", "default"), schema["global_config"].get("backend_host", "http://127.0.0.1:8000"))
                task_copy["full_url"] = urljoin(backend_host, endpoint.lstrip("/"))
            else:
                task_copy["full_url"] = endpoint
        resolved_schema["tasks"] = [t for t in resolved_schema["tasks"] if t["task_id"] != task["task_id"]]
        resolved_schema["tasks"].append(task_copy)

    # Save resolved schema
    output_dir = os.path.join("resolved_schema", f"{schema['pipeline_id']}_{schema['schema_version']}.json")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w") as f:
        json.dump(resolved_schema, f, indent=2)

    # Execute tasks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {}
        for task in tasks:
            if not list(dag.predecessors(task["task_id"])):  # No dependencies
                future = executor.submit(execute_task, task, schema, db)
                future_to_task[future] = task

        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                context[task["task_id"]] = future.result()
                logger.info(f"Task {task['task_id']} executed successfully")
                # Schedule dependent tasks
                for successor in dag.successors(task["task_id"]):
                    successor_task = next(t for t in tasks if t["task_id"] == successor)
                    if all(pred in context for pred in dag.predecessors(successor)):
                        future = executor.submit(execute_task, successor_task, schema, db)
                        future_to_task[future] = successor_task
            except Exception as e:
                logger.error(f"Task {task['task_id']} failed: {str(e)}")
                if not schema["global_config"]["retry_on_failure"]:
                    raise
                for _ in range(schema["global_config"]["max_retries"]):
                    try:
                        context[task["task_id"]] = execute_task(task, schema, db)
                        logger.info(f"Task {task['task_id']} retry succeeded")
                        break
                    except Exception as retry_e:
                        logger.error(f"Retry failed for task {task['task_id']}: {str(retry_e)}")
                        time.sleep(2)
                else:
                    raise

    logger.info(f"Pipeline {schema['pipeline_id']} execution completed")

def execute_single_task(client: Any, db_name: str, schema: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single task via API request."""
    db = client[db_name]
    return execute_task(task, schema, db)

if __name__ == "__main__":
    client = get_mongo_client()
    db = client["AgentBI-Demo"]
    with open("/Users/mohammednihal/Desktop/Business Intelligence/AgentBI/Backend/schemas/AgentBI-Demo.json", "r") as f:
        schema = json.load(f)
    execute_pipeline(client, "AgentBI-Demo", schema)