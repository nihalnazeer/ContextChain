from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)

_task_handlers: Dict[str, Callable] = {}

def register_task(task_type: str, handler: Callable) -> None:
    """
    Register a custom task handler.

    Args:
        task_type (str): Type of the task.
        handler (Callable): Function to handle the task.
    """
    try:
        _task_handlers[task_type] = handler
        logger.info(f"✓ Registered task type: {task_type}")
    except Exception as e:
        logger.error(f"✗ Failed to register task {task_type}: {e}")
        raise

def get_task_handler(task_type: str) -> Callable:
    """
    Get the handler for a task type.

    Args:
        task_type (str): Type of the task.

    Returns:
        Callable: Handler function or None if not found.
    """
    handler = _task_handlers.get(task_type)
    if handler:
        logger.info(f"✓ Retrieved handler for task type: {task_type}")
    else:
        logger.warning(f"No handler found for task type: {task_type}")
    return handler