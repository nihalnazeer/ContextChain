#!/usr/bin/env python3
import click
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from contextchain.engine.validator import validate_schema
from contextchain.engine.executor import execute_pipeline, execute_single_task
from contextchain.registry.version_manager import push_schema, list_versions, rollback_version
from contextchain.registry.schema_loader import load_schema
from contextchain.db.mongo_client import get_mongo_client
from contextchain.db.collections import setup_collections
import subprocess
import os
import time
import stat

# Global variable to track the mongod process
mongod_process = None

# Force colorama initialization for better color support
try:
    import colorama
    colorama.init()
except ImportError:
    pass

def show_banner():
    """Display the CLI banner with ASCII art and colors."""
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
    click.secho("=" * 60, fg="bright_blue", bold=True)
    click.secho(ascii_art, fg="bright_blue", bold=True)
    click.secho("         ContextChain v1.0 CLI", fg="bright_cyan", bold=True)
    click.secho("         Orchestrate AI and Full-Stack Workflows", fg="bright_cyan", bold=True)
    click.secho("                        v1.0", fg="bright_white", bold=True)
    click.secho("=" * 60, fg="bright_blue", bold=True)

def update_current_schema(client, db_name, pipeline_id, version=None):
    """Update the current_schema.json with the latest or specified schema version."""
    schema = load_schema(client, db_name, pipeline_id, version)
    if schema:
        current_schema_path = Path("schemas/current_schema.json")
        current_schema_path.parent.mkdir(exist_ok=True)
        meta = {
            "_meta": {
                "pipeline_id": pipeline_id,
                "version": schema.get("schema_version", "v0.0.0"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "schema": schema
        }
        if not current_schema_path.exists() or os.access(current_schema_path, os.W_OK):
            with current_schema_path.open("w") as f:
                json.dump(meta, f, indent=2)
            current_schema_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            click.secho(f"✓ Updated current_schema.json for {pipeline_id} (version: {schema.get('schema_version', 'v0.0.0')})", fg="bright_green", bold=True)
        else:
            click.secho(f"✓ current_schema.json for {pipeline_id} is read-only, no update performed.", fg="bright_green", bold=True)
    else:
        click.secho(f"✗ No schema found for {pipeline_id}", fg="red", bold=True)

class ColoredGroup(click.Group):
    """Custom Click Group that shows banner and colored help."""
    def format_help(self, ctx, formatter):
        show_banner()
        click.secho("\nUsage: contextchain [OPTIONS] COMMAND [ARGS]...", fg="bright_white", bold=True)
        click.secho("\nContextChain v1.0 CLI", fg="bright_cyan")

        click.secho("\nOptions:", fg="bright_yellow", bold=True)
        click.secho("  -h, --help      Show this help message and exit.", fg="bright_cyan")

        click.secho("\nCommands (grouped by category):", fg="bright_yellow", bold=True)

        click.secho("\n  Initialization:", fg="bright_green", bold=True)
        click.secho("    init          Initialize a new pipeline with a JSON schema and MongoDB setup.", fg="bright_cyan")
        click.secho("                  Options:", fg="bright_cyan")
        click.secho("                    --file PATH          Output path for schema (optional)", fg="bright_cyan")
        click.secho("                    --interactive/--no-interactive  Enable interactive prompts (default: True)", fg="bright_cyan")
        click.secho("                    --backend-host URL   Backend host (e.g., http://127.0.0.1:8000) (optional)", fg="bright_cyan")
        click.secho("                    --frontend-host URL  Frontend host (e.g., http://127.0.0.1:3636) (optional)", fg="bright_cyan")

        click.secho("\n  Schema Management:", fg="bright_green", bold=True)
        click.secho("    schema-compile  Validate a schema file.", fg="bright_cyan")
        click.secho("                    Required: --file PATH", fg="bright_cyan")
        click.secho("    schema-push     Push a schema to MongoDB with versioning.", fg="bright_cyan")
        click.secho("                    Required: --file PATH", fg="bright_cyan")
        click.secho("    schema-pull     Pull a schema from MongoDB.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id", fg="bright_cyan")
        click.secho("                    Optional: --version", fg="bright_cyan")
        click.secho("    schema-version  Create or update a schema version for a pipeline.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id", fg="bright_cyan")
        click.secho("                    Optional: --type (major, minor, patch)", fg="bright_cyan")
        click.secho("    schema-current  Display the current schema version.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id", fg="bright_cyan")
        click.secho("    version-list    List schema versions for a pipeline.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id", fg="bright_cyan")
        click.secho("    version-rollback  Rollback to a previous schema version.", fg="bright_cyan")
        click.secho("                      Required: --pipeline_id, --version", fg="bright_cyan")

        click.secho("\n  Execution:", fg="bright_green", bold=True)
        click.secho("    run            Run an entire pipeline.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id", fg="bright_cyan")
        click.secho("                    Optional: --version", fg="bright_cyan")
        click.secho("                    Optional: --backend-host URL   Override backend host (e.g., http://127.0.0.1:8000)", fg="bright_cyan")
        click.secho("                    Optional: --frontend-host URL  Override frontend host (e.g., http://127.0.0.1:3636)", fg="bright_cyan")
        click.secho("    run-task       Run a single task for development.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id, --task_id", fg="bright_cyan")
        click.secho("                    Optional: --version", fg="bright_cyan")

        click.secho("\n  Collaboration:", fg="bright_green", bold=True)
        click.secho("    ccshare-init    Initialize a .ccshare file for collaborative MongoDB Atlas access.", fg="bright_cyan")
        click.secho("    ccshare-join    Join an existing .ccshare collaboration.", fg="bright_cyan")
        click.secho("                    Required: --uri", fg="bright_cyan")
        click.secho("    ccshare-status  Check the status of the .ccshare configuration.", fg="bright_cyan")

        click.secho("\n  Utilities:", fg="bright_green", bold=True)
        click.secho("    list-pipelines  List all pipelines in MongoDB.", fg="bright_cyan")
        click.secho("    logs            Display logs for a pipeline.", fg="bright_cyan")
        click.secho("                    Required: --pipeline_id", fg="bright_cyan")
        click.secho("    results         Display results for a specific task.", fg="bright_cyan")
        click.secho("                    Required: --task_id", fg="bright_cyan")

        click.secho("\nNotes:", fg="bright_yellow", bold=True)
        click.secho("  - Use 'contextchain COMMAND --help' for detailed options of each command.", fg="bright_cyan")
        click.secho("  - PATH should be a valid file path (e.g., /path/to/schema.json).", fg="bright_cyan")
        click.secho("  - Version numbers follow semantic versioning (e.g., v0.1.0).", fg="bright_cyan")

@click.group(cls=ColoredGroup, context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """ContextChain v1.0 CLI."""
    pass

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
        use_endpoint = click.confirm(click.style(f"Use a custom endpoint for Task {i+1} LLM (otherwise use global config)?", fg="bright_blue"), default=False)
        if use_endpoint:
            task["endpoint"] = click.prompt(click.style(f"Task {i+1} LLM endpoint (e.g., services.custom_llm.generate_summary)", fg="bright_blue"), default=task["endpoint"])
        else:
            task["prompt_template"] = click.prompt(click.style(f"Task {i+1} LLM prompt", fg="bright_blue"), default="")
    
    elif task_type in ["GET", "POST", "PUT"] and interactive:
        if click.confirm(click.style(f"Add input source or inputs for task {i+1}?", fg="bright_blue"), default=False):
            if click.confirm(click.style(f"Use an input source (e.g., URL, DB string)?", fg="bright_blue"), default=False):
                task["input_source"] = click.prompt(click.style(f"Task {i+1} input source", fg="bright_blue"), default="")
            if click.confirm(click.style(f"Add inputs from other tasks?", fg="bright_blue"), default=False):
                input_ids = click.prompt(click.style(f"Task {i+1} input task IDs (comma-separated)", fg="bright_blue"), default="")
                task["inputs"] = [int(x.strip()) for x in input_ids.split(",") if x.strip()]
    elif task_type == "LOCAL" and interactive:
        if click.confirm(click.style(f"Use trigger_logs for task {i+1}?", fg="bright_blue"), default=False):
            task["output_collection"] = "trigger_logs"
    
    if interactive and click.confirm(click.style(f"Add parameters for task {i+1}?", fg="bright_blue"), default=False):
        params_str = click.prompt(click.style(f"Task {i+1} parameters (YAML)", fg="bright_blue"), default="{}")
        try:
            params = yaml.safe_load(params_str)
            task["parameters"] = params or {}
        except yaml.YAMLError:
            click.secho("Invalid YAML, using empty parameters", fg="red")
            task["parameters"] = {}
        if "max_wait_seconds" not in task["parameters"] and click.confirm(click.style(f"Add max_wait_seconds for task {i+1}?", fg="bright_blue"), default=False):
            task["parameters"]["max_wait_seconds"] = click.prompt(click.style("Max wait seconds", fg="bright_blue"), type=int, default=300)
        if "timeout" not in task["parameters"] and click.confirm(click.style(f"Add timeout for task {i+1}?", fg="bright_blue"), default=False):
            task["parameters"]["timeout"] = click.prompt(click.style("Task timeout (seconds)", fg="bright_blue"), type=int, default=30)
    
    if interactive and click.confirm(click.style(f"Add cron schedule for task {i+1}?", fg="bright_blue"), default=False):
        task["cron"] = click.prompt(click.style(f"Task {i+1} cron schedule", fg="bright_blue"), default="")
    
    tasks.append(task)
    return tasks

@cli.command()
@click.option('--file', type=click.Path(), help='Output path for schema')
@click.option('--interactive/--no-interactive', default=True, help='Enable interactive prompts')
@click.option('--backend-host', default="http://127.0.0.1:8000", help='Backend host URL (e.g., http://127.0.0.1:8000)')
@click.option('--frontend-host', default="http://127.0.0.1:3636", help='Frontend host URL (e.g., http://127.0.0.1:3636)')
def init(file, interactive, backend_host, frontend_host):
    """Initialize a new pipeline with a JSON schema and MongoDB setup."""
    global mongod_process
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

    # MongoDB setup with retry loop
    config = {}
    max_attempts = 3
    attempt = 1
    while attempt <= max_attempts:
        mode = click.prompt(click.style(f"MongoDB mode (1: Default (local), 2: .ccshare) (attempt {attempt}/{max_attempts})", fg="bright_blue"), type=click.Choice(["1", "2"]), default="1") if interactive else "1"
        
        if mode == "2":
            ccshare_path = click.prompt(click.style("Path to .ccshare file", fg="bright_blue"), default="config/team.ccshare")
            try:
                with open(ccshare_path, 'r') as f:
                    ccshare = yaml.safe_load(f)
                if not ccshare or "uri" not in ccshare:
                    click.secho(f"✗ Invalid or missing 'uri' in {ccshare_path}. Please check the file.", fg="red", bold=True)
                    if attempt == max_attempts:
                        return
                    continue
                config["uri"] = ccshare["uri"]
                config["db_name"] = ccshare.get("db_name", "contextchain_db")
                config["ccshare_path"] = ccshare_path
            except FileNotFoundError:
                click.secho(f"✗ File not found: {ccshare_path}", fg="red", bold=True)
                if attempt == max_attempts:
                    return
                continue
            except yaml.YAMLError as e:
                click.secho(f"✗ Invalid YAML in {ccshare_path}: {e}", fg="red", bold=True)
                if attempt == max_attempts:
                    return
                continue
        else:
            config["uri"] = f"mongodb://localhost:27017/{pipeline_id}"
            config["db_name"] = pipeline_id

        config_path = Path("config/default_config.yaml")
        config_path.parent.mkdir(exist_ok=True)
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)
        
        click.secho("Setting up MongoDB connection...", fg="bright_yellow")
        try:
            client = get_mongo_client(config["uri"])
            setup_collections(client, config["db_name"])
            click.secho("MongoDB setup completed.", fg="bright_green", bold=True)
            break
        except Exception as e:
            click.secho(f"✗ Failed to connect to MongoDB at {config['uri']}: {str(e)}", fg="red", bold=True)
            if attempt == max_attempts:
                click.secho("✗ Max retry attempts reached. Please fix the configuration and try again.", fg="red", bold=True)
                return
            attempt += 1
            if "mongodb://localhost" in config["uri"]:
                click.secho("Attempting to start a local MongoDB instance...", fg="bright_yellow")
                try:
                    mongod_process = subprocess.Popen(["mongod"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(2)
                    client = get_mongo_client(f"mongodb://localhost:27017/{pipeline_id}")
                    setup_collections(client, pipeline_id)
                    config["uri"] = f"mongodb://localhost:27017/{pipeline_id}"
                    with config_path.open("w") as f:
                        yaml.safe_dump(config, f)
                    click.secho("✓ Local MongoDB instance started and setup completed.", fg="bright_green", bold=True)
                    break
                except Exception as local_e:
                    click.secho(f"✗ Failed to start local MongoDB: {str(local_e)}", fg="red", bold=True)
                    if mongod_process and mongod_process.poll() is None:
                        mongod_process.terminate()
                        try:
                            mongod_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            mongod_process.kill()
                    continue

    if mongod_process and mongod_process.poll() is None and f"mongodb://localhost:27017/{pipeline_id}" not in config["uri"]:
        mongod_process.terminate()
        try:
            mongod_process.wait(timeout=5)
            click.secho("✓ Local MongoDB instance terminated.", fg="bright_green", bold=True)
        except subprocess.TimeoutExpired:
            mongod_process.kill()
            click.secho("✓ Local MongoDB instance forcefully terminated.", fg="bright_green", bold=True)

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

    llm_config = {}
    if any(task["task_type"] == "LLM" for task in tasks) and interactive:
        if click.confirm(click.style("Configure LLM settings for this pipeline?", fg="bright_blue"), default=False):
            llm_config["url"] = click.prompt(click.style("LLM API URL (e.g., https://openrouter.ai/api/v1/chat/completions)", fg="bright_blue"), default="")
            llm_config["api_key_env"] = click.prompt(click.style("Environment variable for API key (e.g., OPENROUTER_API_KEY, leave empty if using direct key)", fg="bright_blue"), default="OPENROUTER_API_KEY", show_default=True)
            llm_config["api_key"] = click.prompt(click.style("Direct API key (optional, leave empty if using env variable)", fg="bright_blue"), default="", show_default=True) if not llm_config["api_key_env"] else ""
            llm_config["model"] = click.prompt(click.style("LLM model (e.g., mistralai/mistral-small-3.2-24b-instruct:free)", fg="bright_blue"), default="mistralai/mistral-small-3.2-24b-instruct:free")
            llm_config["referer"] = click.prompt(click.style("Referer URL (e.g., http://your-site-url.com)", fg="bright_blue"), default="http://your-site-url.com")
            llm_config["title"] = click.prompt(click.style("X-Title (e.g., Your Site Name)", fg="bright_blue"), default="Your Site Name")

    schema = {
        "pipeline_id": pipeline_id,
        "schema_version": "v0.1.0",
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
            "allowed_domains": [],
            "backend_host": backend_host,
            "frontend_host": frontend_host,
            "llm_config": llm_config if llm_config else {}
        },
        "metadata": {
            "tags": tags,
            "pipeline_type": pipeline_type,
            "linked_pipelines": []
        }
    }
    
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

    try:
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        validate_schema(schema)
        push_schema(client, db_name, schema)
        click.secho(f"✓ Initial schema {pipeline_id} validated and pushed to MongoDB.", fg="bright_green", bold=True)
        update_current_schema(client, db_name, pipeline_id)
    except ValueError as e:
        click.secho(f"✗ Validation error during push: {e}", fg="red", bold=True)
        return
    except Exception as e:
        click.secho(f"✗ Push error: {e}", fg="red", bold=True)
        return

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
        push_schema(client, db_name, schema, increment=True)
        click.secho(f"✓ Schema {schema['pipeline_id']} pushed to MongoDB with incremented version.", fg="bright_green", bold=True)
        update_current_schema(client, db_name, schema["pipeline_id"])
    except ValueError as e:
        click.secho(f"✗ Validation error: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Push error: {e}", fg="red", bold=True)
        sys.exit(1)

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
@click.option('--version', help='Schema version (default: latest)')
@click.option('--backend-host', default=None, help='Override backend host URL (e.g., http://127.0.0.1:8000)')
@click.option('--frontend-host', default=None, help='Override frontend host URL (e.g., http://127.0.0.1:3636)')
def run(pipeline_id, version, backend_host, frontend_host):
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

        if backend_host:
            schema["global_config"]["backend_host"] = backend_host
        if frontend_host:
            schema["global_config"]["frontend_host"] = frontend_host

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
        return

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
        update_current_schema(client, db_name, pipeline_id, version)
    except ValueError as e:
        click.secho(f"✗ Rollback error: {e}", fg="red", bold=True)
        return
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        return

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
            client.server_info()
            click.secho(f"✓ Connected to MongoDB", fg="bright_green", bold=True)
            click.secho(f"  Database: {ccshare['db_name']}", fg="bright_cyan")
            click.secho(f"  Roles: {ccshare['roles']}", fg="bright_cyan")
        except Exception as e:
            click.secho(f"✗ Connection error: {e}", fg="red", bold=True)
            return
    else:
        click.secho("✗ No .ccshare file found.", fg="red", bold=True)
        return

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
        return

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
        return

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
            return
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        return

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
        return

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
            else:
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
            click.secho(f"✗ Schema file not added for {pipeline_id}.", fg="red", bold=True)
            return
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        return

@cli.command()
@click.option('--pipeline_id', required=True, help='Pipeline ID')
def schema_current(pipeline_id):
    """Display the current schema version."""
    click.secho(f"\nDisplaying Current Schema for Pipeline {pipeline_id}...", fg="bright_yellow", bold=True)
    try:
        config_path = Path("config/default_config.yaml")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        client = get_mongo_client(config["uri"])
        db_name = config["db_name"]
        update_current_schema(client, db_name, pipeline_id)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        return

if __name__ == "__main__":
    cli()