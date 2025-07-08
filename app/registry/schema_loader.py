def load_schema(client, db_name, pipeline_id, version=None):
    db = client[db_name]
    query = {"pipeline_id": pipeline_id}
    if version:
        query["schema_version"] = version
    return db["schema_registry"].find_one(query)