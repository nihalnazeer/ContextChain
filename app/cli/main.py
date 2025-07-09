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
    """Display the CLI banner with ASCII art and colors."""
    
    # ASCII art for ContextChain
    ascii_art = r"""
 ██████╗ ██████╗ ███╗   ██╗████████╗███████╗██╗  ██╗████████╗
██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
██║     ██║   ██║██╔██╗ ██║   ██║   █████╗   ╚███╔╝    ██║   
██║     ██║   ██║██║╚██╗██║   ██║   ██╔══╝   ██╔██╗    ██║   
╚██████╗╚██████╔╝██║ ╚████║   ██║   ███████╗██╔╝ ██╗   ██║   
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   
                                                              
 ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗
██╔════╝██║  ██║██╔══██╗██║████╗  ██║
██║     ███████║███████║██║██╔██╗ ██║
██║     ██╔══██║██╔══██║██║██║╚██╗██║
╚██████╗██║  ██║██║  ██║██║██║ ╚████║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
    """
    
    # Display the banner
    click.secho("=" * 60, fg="bright_blue", bold=True)
    click.secho(ascii_art, fg="bright_blue", bold=True)
    click.secho("         Orchestrating AI & Full-Stack Workflows", fg="bright_cyan", bold=True)
    click.secho("                        v1.0", fg="bright_white", bold=True)
    click.secho("=" * 60, fg="bright_blue", bold=True)


class ColoredGroup(click.Group):
    """Custom Click Group that shows banner and colored help."""
    
    def format_help(self, ctx, formatter):
        show_banner()
        click.secho("\nAvailable Commands:", fg="bright_yellow", bold=True)
        super().format_help(ctx, formatter)
        click.secho("\nType 'contextchain COMMAND --help' for command details!", fg="bright_cyan")

@click.group(cls=ColoredGroup, context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """ContextChain v1.0 CLI: Orchestrate AI and Full-Stack Workflows."""
    pass

# Helper function to configure tasks dynamically
def configure_task(i, interactive, allowed_task_types, tasks):
    click.secho(f"\nConfiguring Task {i+1}...", fg="bright_yellow", bold=True)
    
    if interactive:
        task_type = click.prompt(
            click.style(f"Task {i+1} type", fg="bright_blue"),
            default="LOCAL",
            type=click.Choice(allowed_task_types),
            show_choices=True
        )
    else:
        task_type = "LOCAL"
    
    task = {
        "task_id": i + 1,
        "description": click.prompt(click.style(f"Task {i+1} description", fg="bright_blue"), default=f"Task {i+1}") if interactive else f"Task {i+1}",
        "task_type": task_type,
        "endpoint": click.prompt(click.style(f"Task {i+1} endpoint", fg="bright_blue"), default="path.to.function") if interactive else "path.to.function",
        "inputs": [],
        "input_source": None,
        "wait_for_input": False,
        "output_collection": "task_results",
        "prompt_template": None,
        "parameters": {},
        "cron": None
    }
    
    if task_type == "LLM" and interactive:
        task["prompt_template"] = click.prompt(click.style(f"Task {i+1} LLM prompt", fg="bright_blue"), default="")
    elif task_type in ["GET", "POST", "PUT"] and interactive:
        if click.confirm(click.style(f"Add input source for task {i+1}?", fg="bright_blue"), default=False):
            task["input_source"] = click.prompt(click.style(f"Task {i+1} input source (e.g., URL, DB string)", fg="bright_blue"), default="")
    elif task_type == "LOCAL" and interactive:
        if click.confirm(click.style(f"Use trigger_logs for task {i+1}?", fg="bright_blue"), default=False):
            task["output_collection"] = "trigger_logs"
    
    if interactive and click.confirm(click.style(f"Add inputs for task {i+1}?", fg="bright_blue"), default=False):
        input_ids = click.prompt(click.style(f"Task {i+1} input task IDs (comma-separated)", fg="bright_blue"), default="")
        task["inputs"] = [int(x.strip()) for x in input_ids.split(",") if x.strip()]
    
    if interactive and not task["input_source"] and click.confirm(click.style(f"Add input source for task {i+1}?", fg="bright_blue"), default=False):
        task["input_source"] = click.prompt(click.style(f"Task {i+1} input source (e.g., URL, DB string)", fg="bright_blue"), default="")
    
    if interactive and click.confirm(click.style(f"Add parameters for task {i+1}?", fg="bright_blue"), default=False):
        params_str = click.prompt(click.style(f"Task {i+1} parameters (YAML)", fg="bright_blue"), default="{}")
        try:
            params = yaml.safe_load(params_str)
            task["parameters"] = params or {}
        except yaml.YAMLError:
            click.secho("Invalid YAML, using empty parameters", fg="red")
            task["parameters"] = {}
        if click.confirm(click.style(f"Add max_wait_seconds for task {i+1}?", fg="bright_blue"), default=False):
            task["parameters"]["max_wait_seconds"] = click.prompt(click.style("Max wait seconds", fg="bright_blue"), type=int, default=300)
        if click.confirm(click.style(f"Add timeout for task {i+1}?", fg="bright_blue"), default=False):
            task["parameters"]["timeout"] = click.prompt(click.style("Task timeout (seconds)", fg="bright_blue"), type=int, default=30)
    
    if interactive and click.confirm(click.style(f"Add cron for task {i+1}?", fg="bright_blue"), default=False):
        task["cron"] = click.prompt(click.style(f"Task {i+1} cron schedule", fg="bright_blue"), default="")
    
    tasks.append(task)
    return tasks

@cli.command()
@click.option('--file', type=click.Path(), help='Output path for schema')
@click.option('--interactive/--no-interactive', default=True, help='Enable interactive prompts')
def init(file, interactive):
    """Initialize a new pipeline with a JSON schema and MongoDB setup."""
    show_banner()
    click.secho("\nInitializing New Pipeline...", fg="bright_yellow", bold=True)

    pipeline_id = click.prompt(click.style("Pipeline ID", fg="bright_blue"), default="new_pipeline") if interactive else "new_pipeline"
    description = click.prompt(click.style("Description", fg="bright_blue"), default="") if interactive else ""
    created_by = click.prompt(click.style("Creator name", fg="bright_blue"), default="user") if interactive else "user"

    # Optional advanced metadata
    if interactive and click.confirm(click.style("Add optional metadata? (tags, pipeline type)", fg="bright_blue"), default=False):
        tags_input = click.prompt(click.style("Tags (comma-separated, e.g., fullstack,llm)", fg="bright_blue"), default="")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        pipeline_type = click.prompt(click.style("Pipeline type", fg="bright_blue"), default="fullstack-ai")
    else:
        tags = []
        pipeline_type = "fullstack-ai"

    # MongoDB setup
    mode = click.prompt(click.style("MongoDB mode (1: Default (local), 2: .ccshare)", fg="bright_blue"), type=click.Choice(["1", "2"]), default="1") if interactive else "1"
    config = {}
    
    if mode == "2":
        ccshare_path = click.prompt(click.style("Path to .ccshare file", fg="bright_blue"), default="config/team.ccshare")
        try:
            with open(ccshare_path, 'r') as f:
                ccshare = yaml.safe_load(f)
            config["uri"] = ccshare["uri"]
            config["db_name"] = ccshare.get("db_name", "contextchain_db")
            config["ccshare_path"] = ccshare_path
        except FileNotFoundError:
            click.secho(f"File not found: {ccshare_path}", fg="red", bold=True)
            return
        except yaml.YAMLError:
            click.secho("Invalid YAML in .ccshare file", fg="red", bold=True)
            return
    else:
        config["uri"] = click.prompt(click.style("MongoDB URI", fg="bright_blue"), default="mongodb://localhost:27017") if interactive else "mongodb://localhost:27017"
        config["db_name"] = click.prompt(click.style("Default MongoDB database", fg="bright_blue"), default="contextchain_db") if interactive else "contextchain_db"

    config_path = Path("config/default_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    
    click.secho("Setting up MongoDB connection...", fg="bright_yellow")
    try:
        client = get_mongo_client(config["uri"])
        setup_collections(client, config["db_name"])
        click.secho("MongoDB setup completed.", fg="bright_green", bold=True)
    except Exception as e:
        click.secho(f"✗ MongoDB setup failed: {e}", fg="red", bold=True)
        return

    # Task configuration
    tasks = []
    if interactive:
        config_method = click.prompt(
            click.style("Configure tasks via (1: CLI, 2: Manual JSON Edit)", fg="bright_blue"),
            type=click.Choice(["1", "2"]),
            default="1"
        )
        if config_method == "1":
            while True:
                tasks = configure_task(len(tasks), interactive, ["GET", "POST", "PUT", "LLM", "LOCAL"], tasks)
                if not click.confirm(click.style("Add another task?", fg="bright_blue"), default=False):
                    break
        else:
            json_path = click.prompt(click.style("Path to JSON file for tasks", fg="bright_blue"), default="tasks.json")
            schema_dir = Path("ccschema")
            schema_dir.mkdir(exist_ok=True)
            full_path = schema_dir / json_path
            if not full_path.exists():
                default_tasks = [{"task_id": 1, "description": "Default Task", "task_type": "LOCAL", "endpoint": "path.to.function"}]
                with full_path.open("w") as f:
                    json.dump({"tasks": default_tasks}, f, indent=2)
                click.secho(f"✓ Created default task file: {full_path}", fg="bright_green", bold=True)
            try:
                with full_path.open("r") as f:
                    tasks_data = json.load(f)
                tasks = tasks_data.get("tasks", [])
            except json.JSONDecodeError:
                click.secho(f"✗ Invalid JSON format in {full_path}", fg="red", bold=True)
                return
    else:
        tasks = [configure_task(1, interactive, ["GET", "POST", "PUT", "LLM", "LOCAL"], [])[0]]

    schema = {
        "pipeline_id": pipeline_id,
        "schema_version": "v1.0.0",
        "description": description,
        "created_by": created_by,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tasks": tasks,
        "global_config": {
            "default_output_db": config["db_name"],
            "logging_level": "INFO",
            "retry_on_failure": True,
            "max_retries": 2,
            "allowed_task_types": ["GET", "POST", "PUT", "LLM", "LOCAL"],
            "allowed_domains": []
        },
        "metadata": {
            "tags": tags,
            "pipeline_type": pipeline_type,
            "linked_pipelines": []
        }
    }
    
    # Optional advanced global config
    if interactive and click.confirm(click.style("Configure advanced settings? (logging, retries, domains)", fg="bright_blue"), default=False):
        logging_level = click.prompt(click.style("Logging level", fg="bright_blue"), default="INFO", type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR"]))
        retry_on_failure = click.confirm(click.style("Retry on failure?", fg="bright_blue"), default=True)
        max_retries = click.prompt(click.style("Max retries", fg="bright_blue"), type=int, default=2)
        domains_input = click.prompt(click.style("Allowed domains (comma-separated)", fg="bright_blue"), default="")
        allowed_domains = [d.strip() for d in domains_input.split(",") if d.strip()]
        
        schema["global_config"].update({
            "logging_level": logging_level,
            "retry_on_failure": retry_on_failure,
            "max_retries": max_retries,
            "allowed_domains": allowed_domains
        })

    schema_path = Path(file) if file else Path(f"schemas/{pipeline_id}.json")
    schema_path.parent.mkdir(exist_ok=True)
    with schema_path.open("w") as f:
        json.dump(schema, f, indent=2)
    
    click.secho(f"✓ Pipeline initialized: {schema_path}", fg="bright_green", bold=True)

@cli.command()
@click.option('--file', type=click.Path(exists=True), required=True, help='Path to schema file')
def schema_compile(file):
    """Validate a schema file."""
    click.secho("\nValidating Schema...", fg="bright_yellow", bold=True)
    try:
        with open(file, 'r') as f:
            schema = json.load(f)
        validate_schema(schema)
        click.secho("✓ Schema validated successfully.", fg="bright_green", bold=True)
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
    click.secho("\nPushing Schema to MongoDB...", fg="bright_yellow", bold=True)
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
        click.secho(f"✓ Schema {schema['pipeline_id']} pushed to MongoDB.", fg="bright_green", bold=True)
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
    click.secho(f"\nRunning Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
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
        click.secho(f"✓ Pipeline {pipeline_id} executed successfully.", fg="bright_green", bold=True)
    except Exception as e:
        click.secho(f"✗ Execution error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--task_id', type=int, required=True, help='Task ID')
@click.option('--version', help='Schema version (default: latest)')
def run_task(pipeline_id, task_id, version):
    """Run a single task for development."""
    click.secho(f"\nRunning Task {task_id} in Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
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
        click.secho(f"✓ Task {task_id} executed successfully.", fg="bright_green", bold=True)
    except Exception as e:
        click.secho(f"✗ Execution error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
def version_list(pipeline_id):
    """List schema versions for a pipeline."""
    click.secho(f"\nListing Versions for Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        versions = list_versions(client, db_name, pipeline_id)
        if not versions:
            click.secho(f"No versions found for {pipeline_id}.", fg="bright_yellow")
            return
        click.secho(f"Found {len(versions)} version(s):", fg="bright_green")
        for v in versions:
            is_latest = " (latest)" if v.get("is_latest", False) else ""
            click.secho(f"  • Version {v['schema_version']}{is_latest}: Created {v['created_at']}", fg="bright_cyan")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--version', required=True, help='Version to rollback to')
def version_rollback(pipeline_id, version):
    """Rollback to a previous schema version."""
    click.secho(f"\nRolling Back Pipeline {pipeline_id} to Version {version}...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        rollback_version(client, db_name, pipeline_id, version)
        click.secho(f"✓ Rolled back {pipeline_id} to version {version}.", fg="bright_green", bold=True)
    except ValueError as e:
        click.secho(f"✗ Rollback error: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
def ccshare_init():
    """Initialize a .ccshare file for collaborative MongoDB Atlas access."""
    click.secho("\nInitializing .ccshare File...", fg="bright_yellow", bold=True)
    ccshare = {
        "uri": click.prompt(click.style("MongoDB Atlas URI", fg="bright_blue"), default="mongodb+srv://user:pass@cluster0.mongodb.net"),
        "db_name": click.prompt(click.style("Database name", fg="bright_blue"), default="contextchain_db"),
        "roles": []
    }
    while click.confirm(click.style("Add a user role?", fg="bright_blue")):
        user = click.prompt(click.style("Username", fg="bright_blue"))
        role = click.prompt(click.style("Role", fg="bright_blue"), default="readOnly", type=click.Choice(["readOnly", "readWrite"]))
        ccshare["roles"].append({"user": user, "role": role})
    output_path = Path("config/team.ccshare")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(ccshare, f)
    click.secho(f"✓ .ccshare file created: {output_path}", fg="bright_green", bold=True)

@cli.command()
@click.option('--uri', required=True, help='MongoDB Atlas URI')
def ccshare_join(uri):
    """Join an existing .ccshare collaboration."""
    click.secho("\nJoining .ccshare Collaboration...", fg="bright_yellow", bold=True)
    ccshare = {
        "uri": uri,
        "db_name": click.prompt(click.style("Database name", fg="bright_blue"), default="contextchain_db"),
        "roles": [{
            "user": click.prompt(click.style("Username", fg="bright_blue")), 
            "role": click.prompt(click.style("Role", fg="bright_blue"), default="readOnly", type=click.Choice(["readOnly", "readWrite"]))
        }]
    }
    output_path = Path("config/team.ccshare")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(ccshare, f)
    click.secho(f"✓ Joined collaboration: {output_path}", fg="bright_green", bold=True)

@cli.command()
def ccshare_status():
    """Check the status of the .ccshare configuration."""
    click.secho("\nChecking .ccshare Status...", fg="bright_yellow", bold=True)
    ccshare_path = Path("config/team.ccshare")
    if ccshare_path.exists():
        try:
            with ccshare_path.open("r") as f:
                ccshare = yaml.safe_load(f)
            client = get_mongo_client(ccshare["uri"])
            client.server_info()  # Test connection
            click.secho(f"✓ Connected to MongoDB", fg="bright_green", bold=True)
            click.secho(f"  Database: {ccshare['db_name']}", fg="bright_cyan")
            click.secho(f"  Roles: {ccshare['roles']}", fg="bright_cyan")
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
    click.secho(f"\nDisplaying Logs for Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        logs = list(client[db_name]["trigger_logs"].find({"pipeline_id": pipeline_id}))
        if logs:
            click.secho(f"Found {len(logs)} log entries:", fg="bright_green")
            for log in logs:
                click.secho(f"  • {log}", fg="bright_blue")
        else:
            click.secho(f"No logs found for {pipeline_id}.", fg="bright_yellow")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--task_id', type=int, required=True, help='Task ID')
def results(task_id):
    """Display results for a specific task."""
    click.secho(f"\nDisplaying Results for Task {task_id}...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        results = list(client[db_name]["task_results"].find({"task_id": task_id}))
        if results:
            click.secho(f"Found {len(results)} result(s):", fg="bright_green")
            for result in results:
                click.secho(f"  • {result}", fg="bright_blue")
        else:
            click.secho(f"No results found for task {task_id}.", fg="bright_yellow")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--version', help='Schema version (default: latest)')
def schema_pull(pipeline_id, version):
    """Pull a schema from MongoDB."""
    click.secho(f"\nPulling Schema for Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
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
            click.secho(f"✓ Schema pulled: {schema_path}", fg="bright_green", bold=True)
        else:
            click.secho(f"✗ Schema {pipeline_id} not found.", fg="red", bold=True)
            sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
def list_pipelines():
    """List all pipelines in MongoDB."""
    click.secho("\nListing All Pipelines...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        pipelines = client[db_name]["schema_registry"].distinct("pipeline_id")
        if pipelines:
            click.secho(f"Found {len(pipelines)} pipeline(s):", fg="bright_green")
            for pipeline in pipelines:
                click.secho(f"  • {pipeline}", fg="bright_cyan")
        else:
            click.secho("No pipelines found.", fg="bright_yellow")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--type', type=click.Choice(['major', 'minor', 'patch']), default='patch', help='Version increment type')
def schema_version(pipeline_id, type):
    """Create or update a schema version for a pipeline."""
    click.secho(f"\nUpdating Schema Version for Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        versions = list_versions(client, db_name, pipeline_id)
        if not versions:
            click.secho(f"✗ No versions found for {pipeline_id}. Starting with v1.0.0.", fg="red", bold=True)
            new_version = "v1.0.0"
        else:
            latest_version = max(versions, key=lambda x: [int(i) for i in x['schema_version'].replace('v', '').split('.')])
            latest_nums = [int(i) for i in latest_version['schema_version'].replace('v', '').split('.')]
            if type == 'major':
                latest_nums[0] += 1
                latest_nums[1] = 0
                latest_nums[2] = 0
            elif type == 'minor':
                latest_nums[1] += 1
                latest_nums[2] = 0
            else:  # patch
                latest_nums[2] += 1
            new_version = f"v{latest_nums[0]}.{latest_nums[1]}.{latest_nums[2]}"
        
        schema_path = Path(f"schemas/{pipeline_id}.json")
        if schema_path.exists():
            with schema_path.open("r") as f:
                schema = json.load(f)
            schema["schema_version"] = new_version
            with schema_path.open("w") as f:
                json.dump(schema, f, indent=2)
            click.secho(f"✓ Updated schema version to {new_version} for {pipeline_id}.", fg="bright_green", bold=True)
        else:
            click.secho(f"✗ Schema file not found for {pipeline_id}.", fg="red", bold=True)
            sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()