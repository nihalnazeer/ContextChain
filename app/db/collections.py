def setup_collections(client, db_name):
    db = client[db_name]
    db["schema_registry"].create_index("pipeline_id")
    db["task_results"].create_index("task_id")
    db["trigger_logs"].create_index("pipeline_id")