import networkx as nx
from typing import List, Dict

def build_dag(tasks: List[Dict]) -> nx.DiGraph:
    """
    Build a DAG from task dependencies.

    Args:
        tasks (List[Dict]): List of task configurations.

    Returns:
        nx.DiGraph: Directed acyclic graph of tasks.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    G = nx.DiGraph()
    for task in tasks:
        G.add_node(task["task_id"])
        for input_id in task.get("inputs", []):
            G.add_edge(input_id, task["task_id"])
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Task dependencies contain a cycle.")
    return G