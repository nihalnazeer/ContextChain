def validate_schema(schema):
    if not schema.get("pipeline_id"):
        raise ValueError("Missing pipeline_id")
    for task in schema.get("tasks", []):
        if task["task_type"] not in schema["global_config"]["allowed_task_types"]:
            raise ValueError(f"Invalid task_type: {task['task_type']}")