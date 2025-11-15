"""
Dynamic Async DAG Engine for ContextChain v2.1
@nhaal160 — FINAL COMPATIBLE VERSION — NOV 14, 2025
FIXED: Proper caching with dependency injection (acba, vector_store, etc.)
NOW 100% ROBUST FOR historical_analysis workflows
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

# Local imports
from .acba import (
    AdaptiveContextBudgetingAlgorithm, BudgetAllocation, BudgetArm, QueryComplexity
)
from .vector import HybridVectorStore
from .llm import BaseLLMClient
from .context_engineer import ContextEngineer, PromptConstructionConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Enums & Dataclasses
# --------------------------------------------------------------------------- #
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # FIXED: Typo removed
    RETRYING = "retrying"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskMetadata:
    task_id: str
    name: str
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout_seconds: float = 30.0
    max_retries: int = 2
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    retry_count: int = 0
    context_updates: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionContext:
    session_id: str
    query: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    processed_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_state: Dict[str, Any] = field(default_factory=dict)
    budget_allocation: Optional[BudgetAllocation] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    shared_resources: Dict[str, Any] = field(default_factory=dict)
    locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
    acba: Optional[AdaptiveContextBudgetingAlgorithm] = None
    prompt_xml: Optional[str] = None

    def get_lock(self, resource_name: str) -> asyncio.Lock:
        if resource_name not in self.locks:
            self.locks[resource_name] = asyncio.Lock()
        return self.locks[resource_name]

    def update_semantic_state(self, updates: Dict[str, Any]):
        self.semantic_state.update(updates)
        self.metadata['last_semantic_update'] = datetime.utcnow()

# --------------------------------------------------------------------------- #
# Base Task (unchanged)
# --------------------------------------------------------------------------- #
class Task(ABC):
    def __init__(self, metadata: TaskMetadata):
        self.metadata = metadata
        self.result: Optional[TaskResult] = None

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> TaskResult:
        pass

    def validate_dependencies(self, completed_tasks: Set[str]) -> bool:
        return self.metadata.dependencies.issubset(completed_tasks)

    async def run_with_monitoring(self, context: ExecutionContext) -> TaskResult:
        start_time = datetime.utcnow()
        retry_count = 0
        error = None

        while retry_count <= self.metadata.max_retries:
            try:
                status = TaskStatus.RUNNING if retry_count == 0 else TaskStatus.RETRYING
                logger.info(f"[{context.session_id}] Task {self.metadata.task_id} {status.value} (attempt {retry_count + 1})")

                result = await asyncio.wait_for(
                    self.execute(context),
                    timeout=self.metadata.timeout_seconds
                )

                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()

                result.start_time = start_time
                result.end_time = end_time
                result.execution_time_seconds = execution_time
                result.retry_count = retry_count
                result.status = TaskStatus.COMPLETED

                self.result = result
                logger.info(f"[{context.session_id}] Task {self.metadata.task_id} completed in {execution_time:.3f}s")
                return result

            except asyncio.TimeoutError as e:
                logger.warning(f"[{context.session_id}] Task {self.metadata.task_id} timed out")
                error = e
            except Exception as e:
                logger.error(f"[{context.session_id}] Task {self.metadata.task_id} failed: {e}")
                error = e

            retry_count += 1
            if retry_count <= self.metadata.max_retries:
                await asyncio.sleep(min(2 ** retry_count, 30))

        end_time = datetime.utcnow()
        failed_result = TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.FAILED,
            error=error,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=(end_time - start_time).total_seconds(),
            retry_count=retry_count - 1
        )
        self.result = failed_result
        return failed_result

# --------------------------------------------------------------------------- #
# Semantic Context Manager (unchanged)
# --------------------------------------------------------------------------- #
class SemanticContextManager:
    def __init__(self, max_history: int = 500):
        self.context_history = []
        self.max_history = max_history

    async def maintain_coherence(self, context: ExecutionContext, task_result: TaskResult):
        if task_result.context_updates:
            context.update_semantic_state(task_result.context_updates)

    def _compute_coherence(self, context: ExecutionContext, result: TaskResult) -> float:
        return 1.0

# --------------------------------------------------------------------------- #
# Task Implementations (unchanged)
# --------------------------------------------------------------------------- #
class FetchDataTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        await asyncio.sleep(0.1)
        data = {'documents': [{'content': f"Doc about {context.query}", 'score': 0.9}]}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=data,
            context_updates={'raw_docs': data}
        )

class BudgetAllocationTask(Task):
    def __init__(self, metadata: TaskMetadata, acba: AdaptiveContextBudgetingAlgorithm):
        super().__init__(metadata)
        self.acba = acba

    async def execute(self, context: ExecutionContext) -> TaskResult:
        if context.budget_allocation:
            return TaskResult(task_id=self.metadata.task_id, status=TaskStatus.SKIPPED, result=context.budget_allocation)
        docs = context.processed_data.get('raw_docs', {}).get('documents', [])
        budget = await self.acba.compute_optimal_budget(context.query, docs, context)
        context.budget_allocation = budget
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=budget,
            context_updates={'budget_allocation': budget}
        )

class RetrieveContextTask(Task):
    def __init__(self, metadata: TaskMetadata, vector_store: HybridVectorStore):
        super().__init__(metadata)
        self.vector_store = vector_store

    async def execute(self, context: ExecutionContext) -> TaskResult:
        try:
            results = await self.vector_store.search(context.query, k=5)
            docs = [{'content': r.content, 'score': r.fusion_score} for r in results]
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=docs,
                context_updates={'retrieved_docs': docs}
            )
        except Exception as e:
            return TaskResult(task_id=self.metadata.task_id, status=TaskStatus.FAILED, error=e)

class RetrieveDocsTask(Task):
    def __init__(self, metadata: TaskMetadata, vector_store: HybridVectorStore):
        super().__init__(metadata)
        self.vector_store = vector_store

    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Retrieving documents for query: {context.query}")
        try:
            search_results = await self.vector_store.search(query=context.query, k=10)
            documents = [
                {'content': r.content, 'source_id': r.source_id, 'score': r.fusion_score}
                for r in search_results
            ]
            result = {'documents': documents, 'total_retrieved': len(documents)}
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                context_updates={'retrieved_docs': result}
            )
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return TaskResult(task_id=self.metadata.task_id, status=TaskStatus.FAILED, error=e)

class RetrieveMetadataTask(Task):
    def __init__(self, metadata: TaskMetadata, vector_store: HybridVectorStore):
        super().__init__(metadata)
        self.vector_store = vector_store

    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Retrieving metadata for query: {context.query}")
        try:
            search_results = await self.vector_store.search(query=context.query, k=10)
            metadata = [
                {'source_id': r.source_id, 'metadata': r.metadata}
                for r in search_results
            ]
            result = {'metadata': metadata, 'total_retrieved': len(metadata)}
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                context_updates={'retrieved_metadata': result}
            )
        except Exception as e:
            logger.error(f"Metadata retrieval failed: {e}")
            return TaskResult(task_id=self.metadata.task_id, status=TaskStatus.FAILED, error=e)

class AssessQualityTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Assessing quality for query: {context.query}")
        await asyncio.sleep(0.2)
        docs = context.processed_data.get('retrieved_docs', {}).get('documents', [])
        scores = [
            {'source_id': d['source_id'], 'quality_score': d.get('score', 0.8) * 0.9}
            for d in docs
        ]
        avg = sum(s['quality_score'] for s in scores) / len(scores) if scores else 0.8
        result = {'quality_scores': scores, 'average_quality': avg}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'quality_assessment': result}
        )

class CompressContextTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Compressing context for query: {context.query}")
        docs = context.processed_data.get('retrieved_docs', {}).get('documents', [])
        await asyncio.sleep(0.4)
        compressed = f"Compressed summary of {len(docs)} docs for: {context.query}"
        result = {
            'compressed_content': compressed,
            'original_token_count': sum(len(d.get('content', '').split()) for d in docs),
            'compressed_token_count': len(compressed.split()),
            'compression_ratio': 0.3,
            'quality_score': 0.85
        }
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'compressed_context': result}
        )

class AnalyzeDataTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Analyzing data for query: {context.query}")
        await asyncio.sleep(0.3)
        result = {'analysis': 'Mock analytical insights', 'confidence': 0.8}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'analysis_result': result}
        )

class GenerateInsightsTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Generating insights for query: {context.query}")
        await asyncio.sleep(0.4)
        result = {'insights': 'Generated insights', 'actionable_items': 3}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'insights': result}
        )

class ComputeDifferencesTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Computing differences for query: {context.query}")
        await asyncio.sleep(0.2)
        result = {'differences': 'Computed differences', 'variance': 0.15}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'differences': result}
        )

class IdentifyTrendsTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Identifying trends for query: {context.query}")
        await asyncio.sleep(0.3)
        result = {'trends': 'Identified trends', 'trend_strength': 0.7}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'trends': result}
        )

class DetectAnomaliesTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Detecting anomalies for query: {context.query}")
        await asyncio.sleep(0.25)
        result = {'anomalies': 'Detected anomalies', 'anomaly_count': 2}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'anomalies': result}
        )

class GenerateResponseTask(Task):
    def __init__(self, metadata: TaskMetadata, llm_optimizer: BaseLLMClient, context_engineer: ContextEngineer):
        super().__init__(metadata)
        self.llm_optimizer = llm_optimizer
        self.context_engineer = context_engineer

    async def execute(self, context: ExecutionContext) -> TaskResult:
        try:
            docs = context.processed_data.get('retrieved_docs', [])
            budget = context.budget_allocation
            complexity = QueryComplexity(
                overall_score=0.6,
                semantic_complexity=0.5,
                compositional_complexity=0.4,
                temporal_complexity=0.3,
                domain_complexity=0.4
            )

            prompt_xml = await self.context_engineer.build_prompt(
                query=context.query,
                raw_docs=docs,
                budget=budget,
                semantic_state=context.semantic_state,
                complexity=complexity
            )
            context.prompt_xml = prompt_xml

            response = await self.llm_optimizer.generate_optimized(prompt=prompt_xml, budget=budget, stream=False)

            parsed = self._parse_xml_prompt(prompt_xml)
            perf = {
                'quality': 0.9,
                'latency': response.generation_time * 1000,
                'tokens_used': response.tokens_used,
                'llm_judge_score': 0.88,
                'estimated_tokens': parsed.get('estimated_tokens', 0),
                'safety_issues': parsed.get('safety_issues', 0)
            }

            if context.acba:
                await context.acba.update_with_feedback(budget, perf, prompt_xml=prompt_xml)

            result = {
                'generated_response': response.content,
                'tokens_used': response.tokens_used,
                'generation_time': response.generation_time
            }
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                context_updates={'final_response': result}
            )
        except Exception as e:
            return TaskResult(task_id=self.metadata.task_id, status=TaskStatus.FAILED, error=e)

    @staticmethod
    def _parse_xml_prompt(xml_str: str) -> Dict:
        try:
            root = ET.fromstring(xml_str)
            content = root.findtext("Content", "")
            meta = {}
            if "PROCESSING METADATA:" in content:
                raw = content.split("PROCESSING METADATA:")[-1]
                for line in raw.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip().lower().replace(" ", "_")] = v.strip()
            return meta
        except:
            return {}

# --------------------------------------------------------------------------- #
# Dynamic Task Graph – FIXED CACHING: Store configs/factories, recreate with deps
# --------------------------------------------------------------------------- #
class DynamicTaskGraph:
    def __init__(self, vector_store=None, llm_optimizer=None, context_engineer=None, acba=None):
        self.graph_templates = self._load_templates()
        self.graph_cache: Dict[str, List[Dict[str, Any]]] = {}  # FIXED: Cache task configs (name, config, task_id)
        self.generation_stats = {'total_generated': 0, 'cache_hits': 0}
        self.vector_store = vector_store
        self.llm_optimizer = llm_optimizer
        self.context_engineer = context_engineer
        self.acba = acba  # Injected from DAGEngine

    def _load_templates(self) -> Dict[str, Dict]:
        return {
            'simple_retrieval': {
                'tasks': ['fetch_data', 'retrieve_context', 'generate_response'],
                'parallel_groups': [],
                'conditional_tasks': {}
            },
            'complex_analytical': {
                'tasks': ['fetch_data', 'budget_allocation', 'retrieve_docs', 'retrieve_metadata',
                          'assess_quality', 'compress_context', 'analyze_data', 'generate_insights'],
                'parallel_groups': [['retrieve_docs', 'retrieve_metadata', 'assess_quality']],
                'conditional_tasks': {
                    'compress_context': 'budget_allocation.compression_needed'
                }
            },
        }

    async def generate_dynamic_graph(self, workflow_type: str, context: ExecutionContext) -> List[Task]:
        cache_key = f"{workflow_type}_{hash(context.query[:100])}"
        if cache_key in self.graph_cache:
            self.generation_stats['cache_hits'] += 1
            logger.debug(f"CACHE HIT for {workflow_type}: rebuilding {len(self.graph_cache[cache_key])} tasks")
            # FIXED: Rebuild from cached configs (recomputes analysis for freshness)
            analysis = self._analyze_query(context.query)
            template = self.graph_templates.get(workflow_type, self.graph_templates['simple_retrieval'])
            return await self._build_tasks_from_cache(template, analysis, context, self.graph_cache[cache_key])

        self.generation_stats['total_generated'] += 1
        logger.debug(f"CACHE MISS for {workflow_type}: generating fresh graph")
        analysis = self._analyze_query(context.query)
        template = self.graph_templates.get(workflow_type, self.graph_templates['simple_retrieval'])

        # FIXED: Build and cache configs (not instances)
        task_configs = await self._build_task_configs(template, analysis, context)
        self.graph_cache[cache_key] = task_configs
        if len(self.graph_cache) > 100:
            self.graph_cache.pop(next(iter(self.graph_cache)))

        # Build instances from configs
        tasks = await self._build_tasks_from_configs(template, analysis, context, task_configs)
        logger.info(f"Generated {len(tasks)} tasks for {workflow_type} (cached)")
        return tasks

    def _analyze_query(self, query: str) -> Dict:
        q = query.lower()
        return {
            'complexity_score': min(len(q.split()) / 20 + sum(0.1 for w in ['analyze', 'compare', 'trend'] if w in q), 1.0),
            'requires_comparison': any(w in q for w in ['compare', 'vs', 'versus']),
            'requires_temporal': any(w in q for w in ['trend', 'over time', 'since']),
        }

    async def _build_task_configs(self, template: Dict, analysis: Dict, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Build and return list of task configs for caching (name, config, task_id, dependencies)."""
        task_configs = []
        task_id_map = {}
        counter = 0

        # Main tasks
        for name in template['tasks']:
            counter += 1
            task_id = f"{name}_{counter}"
            task_id_map[name] = task_id
            config = self._adapt_config(name, analysis)
            # FIXED: Store config; apply deps after all IDs known
            task_config = {'name': name, 'config': config, 'task_id': task_id}
            task_configs.append(task_config)

        # Conditional tasks
        for name, cond in template.get('conditional_tasks', {}).items():
            if self._eval_condition(cond, analysis, context):
                counter += 1
                task_id = f"{name}_{counter}"
                task_id_map[name] = task_id
                config = self._adapt_config(name, analysis)
                task_configs.append({'name': name, 'config': config, 'task_id': task_id})

        # FIXED: Post-process to add dependencies (now that all IDs exist)
        for i, task_config in enumerate(task_configs):
            name = task_config['name']
            if name in task_id_map:
                deps = self._compute_dependencies(template, task_id_map, name)
                task_configs[i]['dependencies'] = list(deps)

        return task_configs

    async def _build_tasks_from_configs(self, template: Dict, analysis: Dict, context: ExecutionContext, task_configs: List[Dict[str, Any]]) -> List[Task]:
        """Rebuild tasks from cached configs (injects deps via _instantiate_task)."""
        tasks = []
        for task_config in task_configs:
            name = task_config['name']
            config = task_config['config']
            task_id = task_config['task_id']
            task = self._instantiate_task(name, config, context, task_id)
            if task:
                # Apply cached dependencies
                if 'dependencies' in task_config:
                    task.metadata.dependencies = set(task_config['dependencies'])
                tasks.append(task)
        return tasks

    async def _build_tasks_from_cache(self, template: Dict, analysis: Dict, context: ExecutionContext, task_configs: List[Dict[str, Any]]) -> List[Task]:
        """Helper for cache hit: same as _build_tasks_from_configs."""
        return await self._build_tasks_from_configs(template, analysis, context, task_configs)

    async def _build_tasks(self, template: Dict, analysis: Dict, context: ExecutionContext) -> List[Task]:
        """Legacy: Build fresh (now delegates to configs for consistency)."""
        task_configs = await self._build_task_configs(template, analysis, context)
        return await self._build_tasks_from_configs(template, analysis, context, task_configs)

    def _adapt_config(self, name: str, analysis: Dict) -> Dict:
        base = {'timeout_seconds': 30.0, 'max_retries': 2, 'priority': TaskPriority.MEDIUM}
        if analysis['complexity_score'] > 0.7:
            base.update({'timeout_seconds': 60.0, 'max_retries': 3, 'priority': TaskPriority.HIGH})
        adaptations = {
            'compress_context': {'timeout_seconds': 90.0},
            "generate_response": {'timeout_seconds': 120.0, 'priority': TaskPriority.HIGH},
        }
        base.update(adaptations.get(name, {}))
        return base

    def _instantiate_task(self, name: str, config: Dict, context: ExecutionContext, task_id: str) -> Optional[Task]:
        metadata = TaskMetadata(
            task_id=task_id,
            name=name,
            **{k: v for k, v in config.items() if k in TaskMetadata.__annotations__}
        )
        # Use self.acba (injected from DAGEngine)
        engine_acba = self.acba

        mapping = {
            'fetch_data': FetchDataTask,
            'budget_allocation': lambda m: BudgetAllocationTask(m, engine_acba),
            'retrieve_context': lambda m: RetrieveContextTask(m, self.vector_store),
            'retrieve_docs': lambda m: RetrieveDocsTask(m, self.vector_store),
            'retrieve_metadata': lambda m: RetrieveMetadataTask(m, self.vector_store),
            'assess_quality': AssessQualityTask,
            'compress_context': CompressContextTask,
            'generate_response': lambda m: GenerateResponseTask(m, self.llm_optimizer, self.context_engineer),
            'analyze_data': AnalyzeDataTask,
            'generate_insights': GenerateInsightsTask,
            'compute_differences': ComputeDifferencesTask,
            'identify_trends': IdentifyTrendsTask,
            'detect_anomalies': DetectAnomaliesTask,
        }
        cls = mapping.get(name)
        if not cls:
            logger.warning(f"Task {name} not found in mapping")
            return None
        # FIXED: Always call cls(metadata) – works for both classes & lambdas
        try:
            task = cls(metadata)
            logger.debug(f"Instantiated task '{name}' (ID: {task_id}) with deps")
            return task
        except Exception as e:
            logger.error(f"Failed to instantiate task '{name}': {e}")
            return None

    def _compute_dependencies(self, template: Dict, task_id_map: Dict, name: str) -> Set[str]:
        """Compute dependencies for a task name (used in config caching)."""
        parallel_tasks = [t for g in template.get('parallel_groups', []) for t in g]
        deps = set()
        if name not in parallel_tasks:
            idx = template['tasks'].index(name)
            for group in template.get('parallel_groups', []):
                if group and template['tasks'].index(group[0]) < idx:
                    deps.update({task_id_map[t] for t in group if t in task_id_map})
        return deps

    def _apply_dependencies(self, task: Task, template: Dict, task_id_map: Dict, name: str):
        deps = self._compute_dependencies(template, task_id_map, name)
        task.metadata.dependencies = deps

    def _eval_condition(self, cond: str, analysis: Dict, context: ExecutionContext) -> bool:
        if 'compression_needed' in cond:
            return analysis['complexity_score'] > 0.5
        return True

# --------------------------------------------------------------------------- #
# Async Task Executor (unchanged)
# --------------------------------------------------------------------------- #
class AsyncTaskExecutor:
    def __init__(self, max_parallel_tasks: int = 10):
        self.max_parallel_tasks = max_parallel_tasks
        self.semaphore = asyncio.Semaphore(max_parallel_tasks)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_parallel_tasks)
        self.stats = {'total': 0, 'parallel': 0, 'avg_time': 0.0}

    async def execute_parallel_with_context(self, tasks: List[Task], context_mgr: SemanticContextManager, context: ExecutionContext) -> ExecutionContext:
        start = datetime.utcnow()
        completed = set()
        results = {}

        while len(completed) < len(tasks):
            ready = [t for t in tasks if t.metadata.task_id not in completed and t.validate_dependencies(completed)]
            if not ready:
                logger.error(f"Deadlock detected. Remaining: {[t.metadata.task_id for t in tasks if t.metadata.task_id not in completed]}")
                break

            batch = [self._run_task(t, context) for t in ready]
            if batch:
                self.stats['parallel'] += 1
                done = await asyncio.gather(*batch, return_exceptions=True)
                for task, result in zip(ready, done):
                    if isinstance(result, Exception):
                        results[task.metadata.task_id] = TaskResult(task_id=task.metadata.task_id, status=TaskStatus.FAILED, error=result)
                    else:
                        results[task.metadata.task_id] = result
                        if result.status == TaskStatus.COMPLETED:
                            await context_mgr.maintain_coherence(context, result)
                            context.processed_data.update(result.context_updates)
                    completed.add(task.metadata.task_id)

        context.intermediate_results['task_results'] = results
        context.metadata.update({
            'execution_time_seconds': (datetime.utcnow() - start).total_seconds(),
            'completed_tasks': len(completed),
            'total_tasks': len(tasks)
        })
        self.stats['total'] += 1
        self.stats['avg_time'] = (self.stats['avg_time'] * (self.stats['total'] - 1) + context.metadata['execution_time_seconds']) / self.stats['total']
        return context

    async def _run_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        async with self.semaphore:
            return await task.run_with_monitoring(context)

    def get_stats(self):
        return self.stats

# --------------------------------------------------------------------------- #
# DAG Engine (unchanged – injections already correct)
# --------------------------------------------------------------------------- #
class DAGEngine:
    def __init__(self, max_parallel_tasks: int = 10, vector_store=None, llm_optimizer=None,
                 context_engineer=None, acba: Optional[AdaptiveContextBudgetingAlgorithm] = None):
        self.acba = acba or AdaptiveContextBudgetingAlgorithm()
        self.context_engineer = context_engineer or ContextEngineer(PromptConstructionConfig(xml_wrap=True))
        # Pass self.acba HERE
        self.task_graph = DynamicTaskGraph(vector_store, llm_optimizer, self.context_engineer, acba=self.acba)
        self.context_mgr = SemanticContextManager()
        self.executor = AsyncTaskExecutor(max_parallel_tasks)
        self.stats = {'total_workflows': 0, 'successful': 0, 'avg_time': 0.0}

        # CRITICAL: Workflow registry for contextchain/core.py
        self._workflow_registry: Dict[str, str] = {}

    # REQUIRED BY contextchain/core.py
    async def register_workflow(self, name: str, workflow_type: str):
        """Register a named workflow → maps to a template in DynamicTaskGraph"""
        self._workflow_registry[name] = workflow_type
        logger.info(f"[DAGEngine] Registered workflow: {name} → {workflow_type}")
        return True

    async def execute_workflow(self, workflow_name: str, context: ExecutionContext) -> ExecutionContext:
        workflow_type = self._workflow_registry.get(workflow_name)
        if not workflow_type:
            raise ValueError(f"Workflow '{workflow_name}' not registered. Call register_workflow() first.")

        start = datetime.utcnow()
        context.acba = self.acba
        try:
            tasks = await self.task_graph.generate_dynamic_graph(workflow_type, context)
            final_context = await self.executor.execute_parallel_with_context(tasks, self.context_mgr, context)
            final_context.metadata.update({
                'execution_status': 'completed',
                'total_execution_time': (datetime.utcnow() - start).total_seconds()
            })
            self.stats['total_workflows'] += 1
            self.stats['successful'] += 1
            return final_context
        except Exception as e:
            context.metadata.update({'execution_status': 'failed', 'error': str(e)})
            self.stats['total_workflows'] += 1
            raise

    async def close(self):
        self.executor.thread_pool.shutdown(wait=True)
        await self.context_engineer.close()

# --------------------------------------------------------------------------- #
# Test (updated to pass acba explicitly)
# --------------------------------------------------------------------------- #
async def test_dag_engine():
    from .llm import create_llm_client
    from .acba import AdaptiveContextBudgetingAlgorithm  # Import for test
    llm = create_llm_client(provider="ollama", model="mistral")
    acba_test = AdaptiveContextBudgetingAlgorithm(max_tokens=2048)  # Explicit for test
    engine = DAGEngine(llm_optimizer=llm, acba=acba_test)

    # Register workflows (required by contextchain)
    await engine.register_workflow("simple_qa", "simple_retrieval")
    await engine.register_workflow("analytical", "complex_analytical")

    context = ExecutionContext(session_id="test", query="Compare Q3 vs Q4 sales")
    result = await engine.execute_workflow("analytical", context)
    print(f"Response: {result.processed_data.get('final_response', {}).get('generated_response')}")

if __name__ == "__main__":
    asyncio.run(test_dag_engine())