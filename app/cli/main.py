#!/usr/bin/env python3
import click
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from app.engine.validator import validate_schema
from app.engine.executor import execute_pipeline, execute_single_task
from app.registry.version_manager import push_schema, list_versions, rollback_version
from app.registry.schema_loader import load_schema
from app.db.mongo_client import get_mongo_client
from app.db.collections import setup_collections

# Force colorama initialization for better color support
try:
    import colorama
    colorama.init()
except ImportError:
    pass

def show_banner():
    """Display the CLI banner with colors."""
    click.secho("=====================================", fg="green", bold=True)
    click.secho("      ContextChain v1.0", fg="green", bold=True)
    click.secho("   Orchestrating AI & Full-Stack Workflows", fg="green", bold=True)
    click.secho("=====================================", fg="green", bold=True)

class ColoredGroup(click.Group):
    """Custom Click Group that shows banner and colored help."""
    
    def format_help(self, ctx, formatter):
        show_banner()
        click.secho("\nAvailable Commands:", fg="yellow", bold=True)
        super().format_help(ctx, formatter)
        click.secho("\nType 'contextchain COMMAND --help' for command details!", fg="cyan")

@click.group(cls=ColoredGroup, context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """ContextChain v1.0 CLI: Orchestrate AI and Full-Stack Workflows."""
    pass

# Helper function to reduce task configuration repetition   
def configure_task(i, interactive, allowed_task_types):
    click.secho(f"\nConfiguring Task {i+1}...", fg="yellow", bold=True)
    
    if interactive:
        task_type = click.prompt(
            click.style(f"Task {i+1} type", fg="cyan"),
            default="LOCAL",
            type=click.Choice(allowed_task_types),
            show_choices=True
        )
    else:
        task_type = "LOCAL"
    
    task = {
        "task_id": i + 1,
        "description": click.prompt(click.style(f"Task {i+1} description", fg="cyan"), default=f"Task {i+1}") if interactive else f"Task {i+1}",
        "task_type": task_type,
        "endpoint": click.prompt(click.style(f"Task {i+1} endpoint", fg="cyan"), default="path.to.function") if interactive else "path.to.function",
        "inputs": [],
        "input_source": None,
        "wait_for_input": click.confirm(click.style(f"Wait for inputs/source for task {i+1}?", fg="cyan"), default=True) if interactive else True,
        "output_collection": "task_results",
        "prompt_template": None,
        "parameters": {},
        "cron": None
    }
    
    if task_type == "LLM":
        task["prompt_template"] = click.prompt(click.style(f"Task {i+1} LLM prompt", fg="cyan"), default="") if interactive else ""
        task["output_collection"] = "task_results"
    elif task_type in ["GET", "POST", "PUT"]:
        task["output_collection"] = "task_results"
        if interactive and click.confirm(click.style(f"Add input source for task {i+1}?", fg="cyan"), default=True):
            task["input_source"] = click.prompt(click.style(f"Task {i+1} input source (e.g., URL, DB string)", fg="cyan"), default="")
    elif task_type == "LOCAL" and interactive and click.confirm(click.style(f"Use trigger_logs for task {i+1}?", fg="cyan"), default=False):
        task["output_collection"] = "trigger_logs"
    
    if interactive and click.confirm(click.style(f"Add inputs for task {i+1}?", fg="cyan"), default=False):
        input_ids = click.prompt(click.style(f"Task {i+1} input task IDs (comma-separated)", fg="cyan"), default="")
        task["inputs"] = [int(x.strip()) for x in input_ids.split(",") if x.strip()]
    
    if interactive and not task["input_source"] and click.confirm(click.style(f"Add input source for task {i+1}?", fg="cyan"), default=False):
        task["input_source"] = click.prompt(click.style(f"Task {i+1} input source (e.g., URL, DB string)", fg="cyan"), default="")
    
    if interactive and click.confirm(click.style(f"Add parameters for task {i+1}?", fg="cyan"), default=False):
        params_str = click.prompt(click.style(f"Task {i+1} parameters (YAML)", fg="cyan"), default="{}")
        try:
            params = yaml.safe_load(params_str)
            task["parameters"] = params
        except yaml.YAMLError:
            click.secho("Invalid YAML, using empty parameters", fg="red")
            task["parameters"] = {}
        
        if click.confirm(click.style(f"Add max_wait_seconds for task {i+1}?", fg="cyan"), default=False):
            task["parameters"]["max_wait_seconds"] = click.prompt(click.style("Max wait seconds", fg="cyan"), type=int, default=300)
        if click.confirm(click.style(f"Add timeout for task {i+1}?", fg="cyan"), default=False):
            task["parameters"]["timeout"] = click.prompt(click.style("Task timeout (seconds)", fg="cyan"), type=int, default=30)
    
    if interactive and click.confirm(click.style(f"Add cron for task {i+1}?", fg="cyan"), default=False):
        task["cron"] = click.prompt(click.style(f"Task {i+1} cron schedule", fg="cyan"), default="")
    
    return task

@cli.command()
@click.option('--file', type=click.Path(), help='Output path for schema')
@click.option('--interactive/--no-interactive', default=True, help='Enable interactive prompts')
def init(file, interactive):
    """Initialize a new pipeline with a JSON schema and MongoDB setup."""
    show_banner()
    click.secho("\nInitializing New Pipeline...", fg="yellow", bold=True)

    pipeline_id = click.prompt(click.style("Pipeline ID", fg="cyan"), default="new_pipeline") if interactive else "new_pipeline"
    description = click.prompt(click.style("Description", fg="cyan"), default="") if interactive else ""
    created_by = click.prompt(click.style("Creator name", fg="cyan"), default="user") if interactive else "user"
    task_count = click.prompt(click.style("Number of tasks", fg="cyan"), type=int, default=1) if interactive else 1
    
    if interactive:
        tags_input = click.prompt(click.style("Tags (comma-separated)", fg="cyan"), default="")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    else:
        tags = []
    
    pipeline_type = click.prompt(click.style("Pipeline type", fg="cyan"), default="fullstack-ai") if interactive else "fullstack-ai"
    
    default_output_db = click.prompt(click.style("Default MongoDB database", fg="cyan"), default="contextchain_db") if interactive else "contextchain_db"
    logging_level = click.prompt(click.style("Logging level", fg="cyan"), default="INFO", type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR"])) if interactive else "INFO"
    retry_on_failure = click.confirm(click.style("Retry on failure?", fg="cyan"), default=True) if interactive else True
    max_retries = click.prompt(click.style("Max retries", fg="cyan"), type=int, default=2) if interactive else 2
    
    if interactive:
        allowed_types_input = click.prompt(click.style("Allowed task types (comma-separated)", fg="cyan"), default="GET,POST,PUT,LLM,LOCAL")
        allowed_task_types = [t.strip() for t in allowed_types_input.split(",") if t.strip()]
        domains_input = click.prompt(click.style("Allowed domains (comma-separated)", fg="cyan"), default="")
        allowed_domains = [d.strip() for d in domains_input.split(",") if d.strip()]
    else:
        allowed_task_types = ["GET", "POST", "PUT", "LLM", "LOCAL"]
        allowed_domains = []
    
    mode = click.prompt(click.style("MongoDB mode (1: Default (local), 2: .ccshare)", fg="cyan"), type=click.Choice(["1", "2"]), default="1") if interactive else "1"
    config = {"db_name": default_output_db}
    
    if mode == "2":
        ccshare_path = click.prompt(click.style("Path to .ccshare file", fg="cyan"), default="config/team.ccshare")
        try:
            with open(ccshare_path, 'r') as f:
                ccshare = yaml.safe_load(f)
            config["uri"] = ccshare["uri"]
            config["ccshare_path"] = ccshare_path
        except FileNotFoundError:
            click.secho(f"File not found: {ccshare_path}", fg="red", bold=True)
            return
        except yaml.YAMLError:
            click.secho("Invalid YAML in .ccshare file", fg="red", bold=True)
            return
    else:
        config["uri"] = click.prompt(click.style("MongoDB URI", fg="cyan"), default="mongodb://localhost:27017") if interactive else "mongodb://localhost:27017"
    
    config_path = Path("config/default_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    
    click.secho("Setting up MongoDB connection...", fg="yellow")
    try:
        client = get_mongo_client(config["uri"])
        setup_collections(client, config["db_name"])
        click.secho("MongoDB setup completed.", fg="green", bold=True)
    except Exception as e:
        click.secho(f"MongoDB setup failed: {e}", fg="red", bold=True)
        return

    tasks = [configure_task(i, interactive, allowed_task_types) for i in range(task_count)]
    
    schema = {
        "pipeline_id": pipeline_id,
        "schema_version": "1.0.0",
        "description": description,
        "created_by": created_by,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tasks": tasks,
        "global_config": {
            "default_output_db": default_output_db,
            "logging_level": logging_level,
            "retry_on_failure": retry_on_failure,
            "max_retries": max_retries,
            "allowed_task_types": allowed_task_types,
            "allowed_domains": allowed_domains
        },
        "metadata": {
            "tags": tags,
            "pipeline_type": pipeline_type,
            "linked_pipelines": []
        }
    }
    
    schema_path = Path(file) if file else Path(f"schemas/{pipeline_id}.json")
    schema_path.parent.mkdir(exist_ok=True)
    with schema_path.open("w") as f:
        json.dump(schema, f, indent=2)
    
    click.secho(f"Pipeline initialized: {schema_path}", fg="green", bold=True)

@cli.command()
@click.option('--file', type=click.Path(exists=True), required=True, help='Path to schema file')
def schema_compile(file):
    """Validate a schema file."""
    click.secho("\nValidating Schema...", fg="yellow", bold=True)
    try:
        with open(file, 'r') as f:
            schema = json.load(f)
        validate_schema(schema)
        click.secho("✓ Schema validated successfully.", fg="green", bold=True)
    except ValueError as e:
        click.secho(f"✗ Validation error: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--file', type=click.Path(exists=True), required=True, help='Path to schema file')
def schema_push(file):
    """Push a schema to MongoDB with versioning."""
    click.secho("\nPushing Schema to MongoDB...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        with open(file, 'r') as f:
            schema = json.load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        
        validate_schema(schema)
        push_schema(client, db_name, schema)
        click.secho(f"✓ Schema {schema['pipeline_id']} pushed to MongoDB.", fg="green", bold=True)
    except ValueError as e:
        click.secho(f"✗ Validation error: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Push error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--version', help='Schema version (default: latest)')
def run(pipeline_id, version):
    """Run an entire pipeline."""
    click.secho(f"\nRunning Pipeline {pipeline_id}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        schema = load_schema(client, db_name, pipeline_id, version)
        
        if not schema:
            click.secho(f"✗ Pipeline {pipeline_id} not found.", fg="red", bold=True)
            sys.exit(1)
        
        execute_pipeline(client, db_name, schema)
        click.secho(f"✓ Pipeline {pipeline_id} executed successfully.", fg="green", bold=True)
    except Exception as e:
        click.secho(f"✗ Execution error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--task_id', type=int, required=True, help='Task ID')
@click.option('--version', help='Schema version (default: latest)')
def run_task(pipeline_id, task_id, version):
    """Run a single task for development."""
    click.secho(f"\nRunning Task {task_id} in Pipeline {pipeline_id}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        schema = load_schema(client, db_name, pipeline_id, version)
        
        if not schema:
            click.secho(f"✗ Pipeline {pipeline_id} not found.", fg="red", bold=True)
            sys.exit(1)
        
        task = next((t for t in schema["tasks"] if t["task_id"] == task_id), None)
        if not task:
            click.secho(f"✗ Task {task_id} not found.", fg="red", bold=True)
            sys.exit(1)
        
        execute_single_task(client, db_name, schema, task)
        click.secho(f"✓ Task {task_id} executed successfully.", fg="green", bold=True)
    except Exception as e:
        click.secho(f"✗ Execution error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
def version_list(pipeline_id):
    """List schema versions for a pipeline."""
    click.secho(f"\nListing Versions for Pipeline {pipeline_id}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        versions = list_versions(client, db_name, pipeline_id)
        
        if not versions:
            click.secho(f"No versions found for {pipeline_id}.", fg="yellow")
            return
        
        click.secho(f"Found {len(versions)} version(s):", fg="green")
        for v in versions:
            is_latest = " (latest)" if v.get("is_latest", False) else ""
            click.secho(f"  • Version {v['schema_version']}{is_latest}: Created {v['created_at']}", fg="cyan")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--version', required=True, help='Version to rollback to')
def version_rollback(pipeline_id, version):
    """Rollback to a previous schema version."""
    click.secho(f"\nRolling Back Pipeline {pipeline_id} to Version {version}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        rollback_version(client, db_name, pipeline_id, version)
        click.secho(f"✓ Rolled back {pipeline_id} to version {version}.", fg="green", bold=True)
    except ValueError as e:
        click.secho(f"✗ Rollback error: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
def ccshare_init():
    """Initialize a .ccshare file for collaborative MongoDB Atlas access."""
    click.secho("\nInitializing .ccshare File...", fg="yellow", bold=True)
    
    ccshare = {
        "uri": click.prompt(click.style("MongoDB Atlas URI", fg="cyan"), default="mongodb+srv://user:pass@cluster0.mongodb.net"),
        "db_name": click.prompt(click.style("Database name", fg="cyan"), default="contextchain_db"),
        "roles": []
    }
    
    while click.confirm(click.style("Add a user role?", fg="cyan")):
        user = click.prompt(click.style("Username", fg="cyan"))
        role = click.prompt(click.style("Role", fg="cyan"), default="readOnly", type=click.Choice(["readOnly", "readWrite"]))
        ccshare["roles"].append({"user": user, "role": role})
    
    output_path = Path("config/team.ccshare")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(ccshare, f)
    
    click.secho(f"✓ .ccshare file created: {output_path}", fg="green", bold=True)

@cli.command()
@click.option('--uri', required=True, help='MongoDB Atlas URI')
def ccshare_join(uri):
    """Join an existing .ccshare collaboration."""
    click.secho("\nJoining .ccshare Collaboration...", fg="yellow", bold=True)
    
    ccshare = {
        "uri": uri,
        "db_name": click.prompt(click.style("Database name", fg="cyan"), default="contextchain_db"),
        "roles": [{
            "user": click.prompt(click.style("Username", fg="cyan")), 
            "role": click.prompt(click.style("Role", fg="cyan"), default="readOnly", type=click.Choice(["readOnly", "readWrite"]))
        }]
    }
    
    output_path = Path("config/team.ccshare")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(ccshare, f)
    
    click.secho(f"✓ Joined collaboration: {output_path}", fg="green", bold=True)

@cli.command()
def ccshare_status():
    """Check the status of the .ccshare configuration."""
    click.secho("\nChecking .ccshare Status...", fg="yellow", bold=True)
    
    ccshare_path = Path("config/team.ccshare")
    if ccshare_path.exists():
        try:
            with ccshare_path.open("r") as f:
                ccshare = yaml.safe_load(f)
            
            client = get_mongo_client(ccshare["uri"])
            client.server_info()  # Test connection
            click.secho(f"✓ Connected to MongoDB", fg="green", bold=True)
            click.secho(f"  Database: {ccshare['db_name']}", fg="cyan")
            click.secho(f"  Roles: {ccshare['roles']}", fg="cyan")
        except Exception as e:
            click.secho(f"✗ Connection error: {e}", fg="red", bold=True)
            sys.exit(1)
    else:
        click.secho("✗ No .ccshare file found.", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
def logs(pipeline_id):
    """Display logs for a pipeline."""
    click.secho(f"\nDisplaying Logs for Pipeline {pipeline_id}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        logs = list(client[db_name]["trigger_logs"].find({"pipeline_id": pipeline_id}))
        
        if logs:
            click.secho(f"Found {len(logs)} log entries:", fg="green")
            for log in logs:
                click.secho(f"  • {log}", fg="blue")
        else:
            click.secho(f"No logs found for {pipeline_id}.", fg="yellow")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--task_id', type=int, required=True, help='Task ID')
def results(task_id):
    """Display results for a specific task."""
    click.secho(f"\nDisplaying Results for Task {task_id}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        results = list(client[db_name]["task_results"].find({"task_id": task_id}))
        
        if results:
            click.secho(f"Found {len(results)} result(s):", fg="green")
            for result in results:
                click.secho(f"  • {result}", fg="blue")
        else:
            click.secho(f"No results found for task {task_id}.", fg="yellow")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--version', help='Schema version (default: latest)')
def schema_pull(pipeline_id, version):
    """Pull a schema from MongoDB."""
    click.secho(f"\nPulling Schema for Pipeline {pipeline_id}...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        schema = load_schema(client, db_name, pipeline_id, version)
        
        if schema:
            schema_path = Path(f"schemas/{pipeline_id}.json")
            schema_path.parent.mkdir(exist_ok=True)
            with schema_path.open("w") as f:
                json.dump(schema, f, indent=2)
            click.secho(f"✓ Schema pulled: {schema_path}", fg="green", bold=True)
        else:
            click.secho(f"✗ Schema {pipeline_id} not found.", fg="red", bold=True)
            sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
def list_pipelines():
    """List all pipelines in MongoDB."""
    click.secho("\nListing All Pipelines...", fg="yellow", bold=True)
    
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        pipelines = client[db_name]["schema_registry"].distinct("pipeline_id")
        
        if pipelines:
            click.secho(f"Found {len(pipelines)} pipeline(s):", fg="green")
            for pipeline in pipelines:
                click.secho(f"  • {pipeline}", fg="cyan")
        else:
            click.secho("No pipelines found.", fg="yellow")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()