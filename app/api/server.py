# app/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from app.db.mongo_client import get_mongo_client
from app.registry.schema_loader import load_schema
from app.engine.executor import execute_pipeline, execute_single_task
from pathlib import Path
import yaml
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ContextChain API", description="API for orchestrating AI and full-stack workflows")

# Pydantic models for request validation
class PipelineRunRequest(BaseModel):
    version: Optional[str] = None

class TaskRunRequest(BaseModel):
    version: Optional[str] = None

# MongoDB client
config_path = Path("config/default_config.yaml")
if config_path.exists():
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    uri = config.get("uri", "mongodb://localhost:27017")
    db_name = config.get("db_name", "contextchain_db")
else:
    uri = "mongodb://localhost:27017"
    db_name = "contextchain_db"
client = get_mongo_client(uri)

@app.post("/pipelines/{pipeline_id}/run")
async def run_pipeline(pipeline_id: str, request: PipelineRunRequest):
    """Execute an entire pipeline."""
    logger.info(f"Received request to run pipeline {pipeline_id}")
    schema = load_schema(client, db_name, pipeline_id, request.version)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    try:
        execute_pipeline(client, db_name, schema)
        return JSONResponse(status_code=200, content={"status": "success", "message": f"Pipeline {pipeline_id} executed"})
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipelines/{pipeline_id}/tasks/{task_id}/run")
async def run_task(pipeline_id: str, task_id: int, request: TaskRunRequest):
    """Execute a specific task."""
    logger.info(f"Received request to run task {task_id} in pipeline {pipeline_id}")
    schema = load_schema(client, db_name, pipeline_id, request.version)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    task = next((t for t in schema["tasks"] if t["task_id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    try:
        result = execute_single_task(client, db_name, schema, task)
        return JSONResponse(status_code=200, content={"status": "success", "result": result})
    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipelines/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """Get the status or logs of a pipeline."""
    logger.info(f"Received request for status of pipeline {pipeline_id}")
    schema = load_schema(client, db_name, pipeline_id)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    db = client[db_name]
    logs = list(db["trigger_logs"].find({"pipeline_id": pipeline_id}))
    results = list(db["task_results"].find({"pipeline_id": pipeline_id}))
    return JSONResponse(status_code=200, content={"logs": logs, "results": results})

@app.get("/pipelines/{pipeline_id}/schema")
async def get_schema(pipeline_id: str, version: Optional[str] = None):
    """Get the schema of a pipeline."""
    logger.info(f"Received request for schema of pipeline {pipeline_id}")
    schema = load_schema(client, db_name, pipeline_id, version)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    return JSONResponse(status_code=200, content=schema)

@app.post("/pipelines/{pipeline_id}/version")
async def update_version(pipeline_id: str, type: str = "patch"):
    """Update the schema version of a pipeline."""
    logger.info(f"Received request to update version for pipeline {pipeline_id}")
    schema_path = Path(f"schemas/{pipeline_id}.json")
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail=f"Schema file not found for {pipeline_id}")
    try:
        with schema_path.open("r") as f:
            schema = json.load(f)
        versions = list(load_schema(client, db_name, pipeline_id, None) or [])
        if not versions:
            new_version = "v1.0.0"
        else:
            latest_version = max(versions, key=lambda x: [int(i) for i in x['schema_version'].replace('v', '').split('.')])
            latest_nums = [int(i) for i in latest_version['schema_version'].replace('v', '').split('.')]
            if type == "major":
                latest_nums[0] += 1; latest_nums[1] = 0; latest_nums[2] = 0
            elif type == "minor":
                latest_nums[1] += 1; latest_nums[2] = 0
            else:  # patch
                latest_nums[2] += 1
            new_version = f"v{latest_nums[0]}.{latest_nums[1]}.{latest_nums[2]}"
        schema["schema_version"] = new_version
        with schema_path.open("w") as f:
            json.dump(schema, f, indent=2)
        return JSONResponse(status_code=200, content={"status": "success", "new_version": new_version})
    except Exception as e:
        logger.error(f"Version update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)