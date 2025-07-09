# ContextChain

![GitHub License](https://img.shields.io/github/license/yourusername/contextchain)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyPI Version](https://img.shields.io/pypi/v/contextchain)
![Build Status](https://img.shields.io/badge/build-passing-green) <!-- Update with CI badge if applicable -->

**ContextChain** is an open-source, pipeline-based execution framework designed to orchestrate AI and full-stack workflows. It enables developers to define, manage, and execute complex task chains with support for various task types (e.g., API calls, local processing, LLM tasks) in a structured and versioned manner. Built with Python, it leverages MongoDB for persistence and provides a command-line interface (CLI) for easy interaction.

## Features
- **Pipeline Management**: Define and execute workflows as pipelines with multiple tasks.
- **Task Types**: Supports GET, POST, PUT, LLM, and LOCAL task types with configurable inputs and outputs.
- **Versioning**: Track and rollback schema versions using MongoDB.
- **Extensibility**: Easily extend with custom task endpoints and configurations.
- **CLI Interface**: Manage pipelines interactively or via scripts.
- **Validation**: Ensures schema integrity with built-in validators.

## Installation

### Via PyPI
ContextChain is available on PyPI. Install it with pip:

```bash
pip install contextchain