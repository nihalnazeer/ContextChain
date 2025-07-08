from pymongo import MongoClient
import time

def execute_pipeline(client, db_name, schema):
    db = client[db_name]
    run_id = f"run_{time.time()}"
    for task in schema["tasks"]:
        db[task["output_collection"]].insert_one({
            "run_id": run_id,
            "task_id": task["task_id"],
            "status": "COMPLETED",
            "output": f"Mock output for task {task['task_id']}"
        })
    print(f"Mock execution of pipeline {schema['pipeline_id']} completed.")

def execute_single_task(client, db_name, schema, task):
    db = client[db_name]
    run_id = f"run_{time.time()}"
    db[task["output_collection"]].insert_one({
        "run_id": run_id,
        "task_id": task["task_id"],
        "status": "COMPLETED",
        "output": f"Mock output for task {task['task_id']}"
    })
    print(f"Mock execution of task {task['task_id']} completed.")