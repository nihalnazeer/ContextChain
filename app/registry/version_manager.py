def push_schema(client, db_name, schema):
    db = client[db_name]
    db["schema_registry"].insert_one(schema)

def list_versions(client, db_name, pipeline_id):
    db = client[db_name]
    return list(db["schema_registry"].find({"pipeline_id": pipeline_id}))

def rollback_version(client, db_name, pipeline_id, version):
    db = client[db_name]
    db["schema_registry"].delete_many({"pipeline_id": pipeline_id, "schema_version": {"$ne": version}})