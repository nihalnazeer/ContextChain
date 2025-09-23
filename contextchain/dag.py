"""
Dynamic Async DAG Engine for ContextChain v2.0
Implements: DynTaskMAS dynamic task graphs, semantic context sharing
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import inspect
import json
from .acba import BudgetAllocation, BudgetArm, QueryComplexity
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from abc import ABC, abstractmethod
from .vector import HybridVectorStore
from .llm import BaseLLMClient

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
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
    
    def get_lock(self, resource_name: str) -> asyncio.Lock:
        if resource_name not in self.locks:
            self.locks[resource_name] = asyncio.Lock()
        return self.locks[resource_name]
    
    def update_semantic_state(self, updates: Dict[str, Any]):
        self.semantic_state.update(updates)
        self.metadata['last_semantic_update'] = datetime.utcnow()

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
        
        while retry_count <= self.metadata.max_retries:
            try:
                status = TaskStatus.RUNNING if retry_count == 0 else TaskStatus.RETRYING
                logger.info(f"Executing task {self.metadata.task_id} (attempt {retry_count + 1})")
                
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
                logger.info(f"Task {self.metadata.task_id} completed in {execution_time:.3f}s")
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Task {self.metadata.task_id} timed out (attempt {retry_count + 1})")
                error = TimeoutError(f"Task timed out after {self.metadata.timeout_seconds}s")
            
            except Exception as e:
                logger.error(f"Task {self.metadata.task_id} failed: {str(e)} (attempt {retry_count + 1})")
                error = e
            
            retry_count += 1
            if retry_count <= self.metadata.max_retries:
                wait_time = min(2 ** retry_count, 30)
                await asyncio.sleep(wait_time)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        failed_result = TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.FAILED,
            error=error,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            retry_count=retry_count - 1
        )
        
        self.result = failed_result
        return failed_result

class SemanticContextManager:
    def __init__(self):
        self.context_history = []
        self.coherence_metrics = {}
        self.semantic_links = {}
        
    async def maintain_coherence(self, context: ExecutionContext, task_result: TaskResult):
        if task_result.context_updates:
            context.update_semantic_state(task_result.context_updates)
        
        coherence_score = self._compute_semantic_coherence(context, task_result)
        context.quality_metrics[f'coherence_{task_result.task_id}'] = coherence_score
        
        context_snapshot = {
            'timestamp': datetime.utcnow(),
            'task_id': task_result.task_id,
            'semantic_state_keys': list(context.semantic_state.keys()),
            'coherence_score': coherence_score
        }
        
        self.context_history.append(context_snapshot)
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-500:]
    
    def _compute_semantic_coherence(self, context: ExecutionContext, task_result: TaskResult) -> float:
        semantic_state = context.semantic_state
        if not semantic_state:
            return 1.0
        
        coherence_factors = []
        value_types = [type(v).__name__ for v in semantic_state.values()]
        type_consistency = len(set(value_types)) / len(value_types) if value_types else 1.0
        coherence_factors.append(1.0 - type_consistency)
        
        recent_updates = context.metadata.get('recent_updates', [])
        if len(recent_updates) > 1:
            temporal_coherence = 0.8
            coherence_factors.append(temporal_coherence)
        
        if hasattr(task_result, 'quality_score'):
            task_quality = getattr(task_result, 'quality_score', 0.5)
            coherence_factors.append(task_quality)
        
        return float(sum(coherence_factors) / len(coherence_factors)) if coherence_factors else 0.8

class DynamicTaskGraph:
    def __init__(self, vector_store=None, llm_optimizer=None, context_engineer=None):
        self.graph_templates = self._initialize_templates()
        self.graph_cache = {}
        self.generation_stats = {'total_generated': 0, 'cache_hits': 0}
        self.vector_store = vector_store
        self.llm_optimizer = llm_optimizer
        self.context_engineer = context_engineer
        
    def _initialize_templates(self) -> Dict[str, Dict]:
        return {
            'simple_retrieval': {
                'description': 'Simple query with direct retrieval and generation',
                'tasks': ['fetch_data', 'retrieve_context', 'generate_response'],
                'parallel_groups': [],
                'conditional_tasks': {}
            },
            'complex_analytical': {
                'description': 'Complex analytical query requiring multi-step processing',
                'tasks': ['fetch_data', 'budget_allocation', 'retrieve_docs', 'retrieve_metadata', 'assess_quality', 'compress_context', 'analyze_data', 'generate_insights'],
                'parallel_groups': [['retrieve_docs', 'retrieve_metadata', 'assess_quality']],
                'conditional_tasks': {
                    'compress_context': 'budget_allocation.compression_needed',
                    'analyze_data': 'retrieve_context.success'
                }
            },
            'comparative_analysis': {
                'description': 'Comparative analysis requiring multiple data sources',
                'tasks': ['fetch_baseline', 'fetch_comparison', 'align_data', 'compute_differences', 'generate_comparison'],
                'parallel_groups': [['fetch_baseline', 'fetch_comparison']],
                'conditional_tasks': {}
            },
            'temporal_analysis': {
                'description': 'Time-series or temporal analysis workflow',
                'tasks': ['fetch_historical', 'identify_trends', 'detect_anomalies', 'forecast_future', 'generate_temporal_insights'],
                'parallel_groups': [['identify_trends', 'detect_anomalies']],
                'conditional_tasks': {
                    'forecast_future': 'identify_trends.has_sufficient_data'
                }
            }
        }
    
    async def generate_dynamic_graph(self, workflow_type: str, context: ExecutionContext) -> List[Task]:
        cache_key = f"{workflow_type}_{hash(str(context.query))}"
        if cache_key in self.graph_cache:
            self.generation_stats['cache_hits'] += 1
            logger.info(f"Using cached graph for workflow: {workflow_type}")
            return self.graph_cache[cache_key].copy()
        
        self.generation_stats['total_generated'] += 1
        query_analysis = self._analyze_query_characteristics(context.query)
        
        if workflow_type in self.graph_templates:
            base_template = self.graph_templates[workflow_type]
        else:
            workflow_type = self._infer_workflow_type(query_analysis)
            base_template = self.graph_templates.get(workflow_type, self.graph_templates['simple_retrieval'])
        
        adaptive_tasks = await self._generate_adaptive_tasks(base_template, query_analysis, context)
        self.graph_cache[cache_key] = adaptive_tasks.copy()
        
        if len(self.graph_cache) > 100:
            oldest_keys = list(self.graph_cache.keys())[:20]
            for key in oldest_keys:
                del self.graph_cache[key]
        
        logger.info(f"Generated dynamic graph with {len(adaptive_tasks)} tasks for workflow: {workflow_type}")
        return adaptive_tasks
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        analysis = {
            'length': len(query.split()),
            'complexity_score': 0.0,
            'query_type': 'general',
            'requires_comparison': False,
            'requires_temporal_analysis': False,
            'requires_quantitative_analysis': False,
            'estimated_processing_complexity': 1.0
        }
        
        if any(word in query_lower for word in ['analyze', 'analysis', 'examine', 'study']):
            analysis['query_type'] = 'analytical'
            analysis['complexity_score'] += 0.3
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'contrast']):
            analysis['query_type'] = 'comparative'
            analysis['requires_comparison'] = True
            analysis['complexity_score'] += 0.4
        elif any(word in query_lower for word in ['trend', 'over time', 'historical', 'timeline', 'since', 'until']):
            analysis['query_type'] = 'temporal'
            analysis['requires_temporal_analysis'] = True
            analysis['complexity_score'] += 0.3
        if any(word in query_lower for word in ['how many', 'percentage', 'ratio', 'statistics', 'metrics']):
            analysis['requires_quantitative_analysis'] = True
            analysis['complexity_score'] += 0.2
        complexity_indicators = ['detailed', 'comprehensive', 'thorough', 'in-depth', 'complete']
        complexity_boost = sum(0.1 for indicator in complexity_indicators if indicator in query_lower)
        analysis['complexity_score'] += complexity_boost
        if any(connector in query_lower for connector in ['and then', 'also', 'furthermore', 'additionally']):
            analysis['complexity_score'] += 0.2
        analysis['estimated_processing_complexity'] = min(analysis['complexity_score'] * 2 + 1, 3.0)
        return analysis
    
    def _infer_workflow_type(self, query_analysis: Dict[str, Any]) -> str:
        if query_analysis['requires_comparison']:
            return 'comparative_analysis'
        elif query_analysis['requires_temporal_analysis']:
            return 'temporal_analysis'
        elif query_analysis['complexity_score'] > 0.5:
            return 'complex_analytical'
        else:
            return 'simple_retrieval'
    
    async def _generate_adaptive_tasks(self, base_template: Dict, query_analysis: Dict[str, Any], context: ExecutionContext) -> List[Task]:
        tasks = []
        task_id_counter = 0
        task_id_map = {}  # Maps task_name to task_id
        
        # Process sequential tasks
        for task_name in base_template['tasks']:
            task_id_counter += 1
            task_id = f"{task_name}_{task_id_counter}"
            task_id_map[task_name] = task_id
            task_config = self._get_adaptive_task_config(task_name, query_analysis, context)
            task = self._create_task_instance(task_name, task_config, context, task_id)
            if task:
                # Add dependencies for tasks following parallel groups
                if 'parallel_groups' in base_template:
                    parallel_tasks = [t for group in base_template['parallel_groups'] for t in group]
                    if task_name not in parallel_tasks:
                        task_index = base_template['tasks'].index(task_name)
                        for group in base_template['parallel_groups']:
                            if group and base_template['tasks'].index(group[0]) < task_index:
                                task.metadata.dependencies.update(
                                    {task_id_map[t] for t in group if t in task_id_map}
                                )
                tasks.append(task)
                logger.debug(f"Added task {task_id} with dependencies {task.metadata.dependencies}")
        
        # Process conditional tasks
        for task_name, condition in base_template.get('conditional_tasks', {}).items():
            if self._evaluate_task_condition(condition, query_analysis, context):
                task_id_counter += 1
                task_id = f"{task_name}_{task_id_counter}"
                task_id_map[task_name] = task_id
                task_config = self._get_adaptive_task_config(task_name, query_analysis, context)
                task = self._create_task_instance(task_name, task_config, context, task_id)
                if task:
                    tasks.append(task)
                    logger.debug(f"Added conditional task {task_id} with dependencies {task.metadata.dependencies}")
        
        return tasks
    
    def _get_adaptive_task_config(self, task_name: str, query_analysis: Dict, context: ExecutionContext) -> Dict:
        base_config = {
            'timeout_seconds': 30.0,
            'max_retries': 2,
            'priority': TaskPriority.MEDIUM
        }
        if query_analysis['estimated_processing_complexity'] > 2.0:
            base_config['timeout_seconds'] = 60.0
            base_config['max_retries'] = 3
            base_config['priority'] = TaskPriority.HIGH
        elif query_analysis['estimated_processing_complexity'] < 1.2:
            base_config['timeout_seconds'] = 15.0
            base_config['max_retries'] = 1
        task_adaptations = {
            'fetch_data': {
                'timeout_seconds': base_config['timeout_seconds'] * 0.5,
                'description': 'Fetch raw data from sources'
            },
            'retrieve_context': {
                'timeout_seconds': base_config['timeout_seconds'] * 1.5,
                'description': 'Retrieve and filter contextual documents'
            },
            'retrieve_docs': {
                'timeout_seconds': base_config['timeout_seconds'] * 1.2,
                'description': 'Retrieve document content for analysis'
            },
            'retrieve_metadata': {
                'timeout_seconds': base_config['timeout_seconds'] * 1.0,
                'description': 'Retrieve metadata for documents'
            },
            'assess_quality': {
                'timeout_seconds': base_config['timeout_seconds'] * 1.3,
                'description': 'Assess quality of retrieved documents'
            },
            'compress_context': {
                'timeout_seconds': base_config['timeout_seconds'] * 2.0,
                'description': 'Compress context using RL optimization'
            },
            'generate_response': {
                'timeout_seconds': base_config['timeout_seconds'] * 2.5,
                'priority': TaskPriority.HIGH,
                'description': 'Generate final response using LLM'
            }
        }
        if task_name in task_adaptations:
            base_config.update(task_adaptations[task_name])
        return base_config
    
    def _create_task_instance(self, task_name: str, config: Dict, context: ExecutionContext, task_id: str) -> Optional[Task]:
        metadata = TaskMetadata(
            task_id=task_id,
            name=task_name,
            description=config.get('description', f'Execute {task_name}'),
            priority=config.get('priority', TaskPriority.MEDIUM),
            timeout_seconds=config.get('timeout_seconds', 30.0),
            max_retries=config.get('max_retries', 2)
        )
        task_classes = {
            'fetch_data': FetchDataTask,
            'budget_allocation': BudgetAllocationTask,
            'retrieve_context': lambda metadata: RetrieveContextTask(metadata, self.vector_store),
            'retrieve_docs': lambda metadata: RetrieveDocsTask(metadata, self.vector_store),
            'retrieve_metadata': lambda metadata: RetrieveMetadataTask(metadata, self.vector_store),
            'assess_quality': AssessQualityTask,
            'compress_context': CompressContextTask,
            'generate_response': lambda metadata: GenerateResponseTask(metadata, self.llm_optimizer, self.context_engineer),
            'analyze_data': AnalyzeDataTask,
            'generate_insights': GenerateInsightsTask,
            'compute_differences': ComputeDifferencesTask,
            'identify_trends': IdentifyTrendsTask,
            'detect_anomalies': DetectAnomaliesTask
        }
        task_class = task_classes.get(task_name)
        if task_class:
            logger.debug(f"Creating task instance for {task_name} with ID {task_id}")
            return task_class(metadata)
        else:
            logger.warning(f"Unknown task type: {task_name}")
            return None
    
    def _evaluate_task_condition(self, condition: str, query_analysis: Dict, context: ExecutionContext) -> bool:
        if 'compression_needed' in condition:
            return query_analysis.get('complexity_score', 0) > 0.3
        elif 'has_sufficient_data' in condition:
            return True
        elif 'success' in condition:
            return True
        return False

class AsyncTaskExecutor:
    def __init__(self, max_parallel_tasks: int = 10):
        self.max_parallel_tasks = max_parallel_tasks
        self.execution_semaphore = asyncio.Semaphore(max_parallel_tasks)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_parallel_tasks)
        self.execution_stats = {
            'total_executions': 0,
            'parallel_executions': 0,
            'avg_execution_time': 0.0
        }
        
    async def execute_parallel_with_context(self, tasks: List[Task], context_manager: SemanticContextManager, initial_context: ExecutionContext) -> ExecutionContext:
        start_time = datetime.utcnow()
        logger.info(f"Starting execution of {len(tasks)} tasks")
        
        dependency_graph = self._build_dependency_graph(tasks)
        completed_tasks = set()
        task_results = {}
        context = initial_context
        
        while len(completed_tasks) < len(tasks):
            ready_tasks = [
                task for task in tasks 
                if (task.metadata.task_id not in completed_tasks and 
                    task.validate_dependencies(completed_tasks))
            ]
            
            if not ready_tasks:
                remaining_tasks = [t for t in tasks if t.metadata.task_id not in completed_tasks]
                logger.error(f"No ready tasks found. Remaining: {[t.metadata.task_id for t in remaining_tasks]}")
                logger.debug(f"Completed tasks: {completed_tasks}")
                logger.debug(f"All task dependencies: {[(t.metadata.task_id, t.metadata.dependencies) for t in tasks]}")
                break
            
            parallel_tasks = []
            for task in ready_tasks:
                parallel_tasks.append(self._execute_task_with_semaphore(task, context))
            
            if parallel_tasks:
                self.execution_stats['parallel_executions'] += 1
                batch_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                
                for i, result in enumerate(batch_results):
                    task = ready_tasks[i]
                    if isinstance(result, Exception):
                        logger.error(f"Task {task.metadata.task_id} failed with exception: {result}")
                        task_result = TaskResult(
                            task_id=task.metadata.task_id,
                            status=TaskStatus.FAILED,
                            error=result
                        )
                    else:
                        task_result = result
                    task_results[task.metadata.task_id] = task_result
                    completed_tasks.add(task.metadata.task_id)
                    if task_result.status == TaskStatus.COMPLETED:
                        await context_manager.maintain_coherence(context, task_result)
                        if task_result.context_updates:
                            context.processed_data.update(task_result.context_updates)
            
            logger.info(f"Completed {len(completed_tasks)}/{len(tasks)} tasks")
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        self.execution_stats['total_executions'] += 1
        self.execution_stats['avg_execution_time'] = (
            (self.execution_stats['avg_execution_time'] * (self.execution_stats['total_executions'] - 1) + 
             execution_time) / self.execution_stats['total_executions']
        )
        
        context.intermediate_results['task_results'] = task_results
        context.metadata['execution_time_seconds'] = execution_time
        context.metadata['completed_tasks'] = len(completed_tasks)
        context.metadata['total_tasks'] = len(tasks)
        
        logger.info(f"DAG execution completed in {execution_time:.3f}s. Success rate: {len(completed_tasks)}/{len(tasks)}")
        return context
    
    async def _execute_task_with_semaphore(self, task: Task, context: ExecutionContext) -> TaskResult:
        async with self.execution_semaphore:
            return await task.run_with_monitoring(context)
    
    def _build_dependency_graph(self, tasks: List[Task]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for task in tasks:
            graph.add_node(task.metadata.task_id, task=task)
        for task in tasks:
            for dependency in task.metadata.dependencies:
                if dependency in [t.metadata.task_id for t in tasks]:
                    graph.add_edge(dependency, task.metadata.task_id)
                else:
                    logger.warning(f"Unresolved dependency {dependency} for task {task.metadata.task_id}")
        return graph
    
    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            'max_parallel_tasks': self.max_parallel_tasks,
            'total_executions': self.execution_stats['total_executions'],
            'parallel_executions': self.execution_stats['parallel_executions'],
            'avg_execution_time': self.execution_stats['avg_execution_time'],
            'current_semaphore_value': self.execution_semaphore._value
        }

class FetchDataTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Fetching data for query: {context.query}")
        await asyncio.sleep(0.1)
        fetched_data = {
            'documents': [
                {'content': f'Mock document 1 for: {context.query}', 'score': 0.9},
                {'content': f'Mock document 2 for: {context.query}', 'score': 0.8}
            ],
            'metadata': {'source': 'mock_db', 'fetch_time': datetime.utcnow()}
        }
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=fetched_data,
            context_updates={'fetched_data': fetched_data}
        )

class BudgetAllocationTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        if context.budget_allocation:
            logger.info(f"Budget already allocated for {context.query}, skipping task")
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.SKIPPED,
                result=context.budget_allocation,
                context_updates={}
            )
        logger.info(f"Computing budget allocation for query: {context.query}")
        await asyncio.sleep(0.2)
        budget_allocation = BudgetAllocation(
            retrieval_tokens=800,
            compression_tokens=400,
            generation_tokens=800,
            total_budget=2000,
            arm_selected=BudgetArm.ADAPTIVE_COMPRESS,
            confidence_score=0.75,
            hierarchy_weights={},
            expected_utility=0.75,
            allocation_timestamp=datetime.utcnow()
        )
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=budget_allocation,
            context_updates={'budget_allocation': budget_allocation}
        )

class RetrieveContextTask(Task):
    def __init__(self, metadata: TaskMetadata, vector_store: HybridVectorStore):
        super().__init__(metadata)
        self.vector_store = vector_store

    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Retrieving context for query: {context.query}")
        try:
            search_results = await self.vector_store.search(query=context.query, k=5)
            filtered_context = [
                {
                    'content': result.content,
                    'source_id': result.source_id,
                    'score': result.fusion_score,
                    'metadata': result.metadata,
                    'embedding_quality': result.embedding_quality
                } for result in search_results
            ]
            context_result = {
                'filtered_documents': filtered_context,
                'total_documents': len(filtered_context),
                'filtered_count': len(filtered_context),
                'quality_threshold': 0.7
            }
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=context_result,
                context_updates={'retrieved_context': context_result}
            )
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.FAILED,
                error=e
            )

class RetrieveDocsTask(Task):
    def __init__(self, metadata: TaskMetadata, vector_store: HybridVectorStore):
        super().__init__(metadata)
        self.vector_store = vector_store

    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Retrieving documents for query: {context.query}")
        try:
            search_results = await self.vector_store.search(query=context.query, k=10)
            documents = [
                {
                    'content': result.content,
                    'source_id': result.source_id,
                    'score': result.fusion_score
                } for result in search_results
            ]
            result = {
                'documents': documents,
                'total_retrieved': len(documents)
            }
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                context_updates={'retrieved_docs': result}
            )
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.FAILED,
                error=e
            )

class RetrieveMetadataTask(Task):
    def __init__(self, metadata: TaskMetadata, vector_store: HybridVectorStore):
        super().__init__(metadata)
        self.vector_store = vector_store

    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Retrieving metadata for query: {context.query}")
        try:
            search_results = await self.vector_store.search(query=context.query, k=10)
            metadata = [
                {
                    'source_id': result.source_id,
                    'metadata': result.metadata
                } for result in search_results
            ]
            result = {
                'metadata': metadata,
                'total_retrieved': len(metadata)
            }
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                context_updates={'retrieved_metadata': result}
            )
        except Exception as e:
            logger.error(f"Metadata retrieval failed: {str(e)}")
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.FAILED,
                error=e
            )

class AssessQualityTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Assessing quality for query: {context.query}")
        await asyncio.sleep(0.2)
        retrieved_docs = context.processed_data.get('retrieved_docs', {}).get('documents', [])
        quality_scores = [
            {'source_id': doc['source_id'], 'quality_score': doc.get('score', 0.8) * 0.9}
            for doc in retrieved_docs
        ]
        result = {
            'quality_scores': quality_scores,
            'average_quality': sum(score['quality_score'] for score in quality_scores) / len(quality_scores) if quality_scores else 0.8
        }
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            context_updates={'quality_assessment': result}
        )

class CompressContextTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Compressing context for query: {context.query}")
        retrieved_docs = context.processed_data.get('retrieved_docs', {}).get('documents', [])
        budget_allocation = context.budget_allocation
        await asyncio.sleep(0.4)
        compressed_content = f"Compressed summary of {len(retrieved_docs)} documents for query: {context.query}"
        compression_result = {
            'compressed_content': compressed_content,
            'original_token_count': sum(len(doc.get('content', '').split()) for doc in retrieved_docs),
            'compressed_token_count': len(compressed_content.split()),
            'compression_ratio': 0.3,
            'quality_score': 0.85
        }
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=compression_result,
            context_updates={'compressed_context': compression_result}
        )

class GenerateResponseTask(Task):
    def __init__(self, metadata: TaskMetadata, llm_optimizer: BaseLLMClient, context_engineer):
        super().__init__(metadata)
        self.llm_optimizer = llm_optimizer
        self.context_engineer = context_engineer

    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Generating response for query: {context.query}")
        try:
            retrieved_docs = context.processed_data.get('retrieved_docs', {}).get('documents', [])
            budget = context.budget_allocation
            complexity = QueryComplexity(
                overall_score=0.5,
                semantic_complexity=0.4,
                compositional_complexity=0.3,
                temporal_complexity=0.2,
                domain_complexity=0.3
            )
            logger.debug(f"Query complexity for {context.query}: {asdict(complexity)}")
            prompt = await self.context_engineer.build_prompt(
                query=context.query,
                raw_docs=retrieved_docs,
                budget=budget,
                semantic_state=context.semantic_state,
                complexity=complexity
            )
            llm_response = await self.llm_optimizer.generate_optimized(
                prompt=prompt,
                budget=budget,
                stream=False
            )
            generation_result = {
                'generated_response': llm_response.content,
                'tokens_used': llm_response.tokens_used,
                'generation_time': llm_response.generation_time,
                'quality_metrics': {
                    'coherence': 0.9,
                    'relevance': 0.85,
                    'completeness': 0.8
                }
            }
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.COMPLETED,
                result=generation_result,
                context_updates={'final_response': generation_result}
            )
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return TaskResult(
                task_id=self.metadata.task_id,
                status=TaskStatus.FAILED,
                error=e
            )

class AnalyzeDataTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Analyzing data for query: {context.query}")
        await asyncio.sleep(0.3)
        analysis_result = {'analysis': 'Mock analytical insights', 'confidence': 0.8}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=analysis_result,
            context_updates={'analysis_result': analysis_result}
        )

class GenerateInsightsTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Generating insights for query: {context.query}")
        await asyncio.sleep(0.4)
        insights = {'insights': 'Generated insights based on analysis', 'actionable_items': 3}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=insights,
            context_updates={'insights': insights}
        )

class ComputeDifferencesTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Computing differences for query: {context.query}")
        await asyncio.sleep(0.2)
        differences = {'differences': 'Computed differences between datasets', 'variance': 0.15}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=differences,
            context_updates={'differences': differences}
        )

class IdentifyTrendsTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Identifying trends for query: {context.query}")
        await asyncio.sleep(0.3)
        trends = {'trends': 'Identified temporal trends', 'trend_strength': 0.7}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=trends,
            context_updates={'trends': trends}
        )

class DetectAnomaliesTask(Task):
    async def execute(self, context: ExecutionContext) -> TaskResult:
        logger.info(f"Detecting anomalies for query: {context.query}")
        await asyncio.sleep(0.25)
        anomalies = {'anomalies': 'Detected anomalies in data', 'anomaly_count': 2}
        return TaskResult(
            task_id=self.metadata.task_id,
            status=TaskStatus.COMPLETED,
            result=anomalies,
            context_updates={'anomalies': anomalies}
        )

class DAGEngine:
    def __init__(self, max_parallel_tasks: int = 10, vector_store=None, llm_optimizer=None, context_engineer=None, templates_dir: Optional[str] = None):
        if context_engineer is None:
            from .context_engineer import ContextEngineer, PromptConstructionConfig
            context_engineer = ContextEngineer(templates_dir=templates_dir)
        self.task_graph_generator = DynamicTaskGraph(vector_store, llm_optimizer, context_engineer)
        self.context_manager = SemanticContextManager()
        self.executor = AsyncTaskExecutor(max_parallel_tasks)
        self.workflow_registry = {
            'historical_analysis': 'complex_analytical',
            'sales_comparison': 'comparative_analysis',
            'trend_analysis': 'temporal_analysis',
            'simple_qa': 'simple_retrieval'
        }
        self.engine_stats = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'avg_workflow_time': 0.0,
            'total_tasks_executed': 0
        }
        logger.info(f"DAG Engine initialized with max {max_parallel_tasks} parallel tasks")
    
    async def execute_workflow(self, workflow_name: str, initial_context: ExecutionContext) -> ExecutionContext:
        start_time = datetime.utcnow()
        workflow_type = self.workflow_registry.get(workflow_name, 'simple_retrieval')
        logger.info(f"Executing workflow: {workflow_name} (type: {workflow_type})")
        
        try:
            # Validate workflow before execution
            validation = await self.validate_workflow(workflow_name)
            if not validation['valid']:
                raise Exception(f"Workflow validation failed: {validation['error']}")
            
            tasks = await self.task_graph_generator.generate_dynamic_graph(workflow_type, initial_context)
            if not tasks:
                raise Exception(f"No tasks generated for workflow: {workflow_name}")
            
            final_context = await self.executor.execute_parallel_with_context(tasks, self.context_manager, initial_context)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            self.engine_stats['total_workflows'] += 1
            self.engine_stats['successful_workflows'] += 1
            self.engine_stats['total_tasks_executed'] += len(tasks)
            self.engine_stats['avg_workflow_time'] = (
                (self.engine_stats['avg_workflow_time'] * (self.engine_stats['total_workflows'] - 1) + 
                 execution_time) / self.engine_stats['total_workflows']
            )
            
            final_context.metadata.update({
                'workflow_name': workflow_name,
                'workflow_type': workflow_type,
                'execution_status': 'completed',
                'total_execution_time': execution_time,
                'tasks_executed': len(tasks)
            })
            
            logger.info(f"Workflow {workflow_name} completed successfully in {execution_time:.3f}s")
            return final_context
            
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            self.engine_stats['total_workflows'] += 1
            error_context = initial_context
            error_context.metadata.update({
                'workflow_name': workflow_name,
                'workflow_type': workflow_type,
                'execution_status': 'failed',
                'error': str(e),
                'total_execution_time': execution_time
            })
            logger.error(f"Workflow {workflow_name} failed: {str(e)}")
            raise Exception(f"Workflow execution failed: {str(e)}") from e
    
    async def register_workflow(self, workflow_name: str, workflow_type: str, custom_template: Optional[Dict] = None):
        if custom_template:
            self.task_graph_generator.graph_templates[workflow_type] = custom_template
        self.workflow_registry[workflow_name] = workflow_type
        logger.info(f"Registered workflow: {workflow_name} -> {workflow_type}")
    
    async def close(self):
        try:
            self.executor.thread_pool.shutdown(wait=True)
            await self.task_graph_generator.context_engineer.close()
            logger.info("DAGEngine thread pool and context engineer shut down")
        except Exception as e:
            logger.error(f"Error closing DAGEngine: {str(e)}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        return {
            'engine_stats': self.engine_stats,
            'executor_stats': self.executor.get_execution_stats(),
            'graph_generator_stats': self.task_graph_generator.generation_stats,
            'workflow_registry': self.workflow_registry,
            'context_coherence_history': len(self.context_manager.context_history)
        }
    
    async def validate_workflow(self, workflow_name: str) -> Dict[str, Any]:
        if workflow_name not in self.workflow_registry:
            return {'valid': False, 'error': f'Workflow {workflow_name} not registered'}
        workflow_type = self.workflow_registry[workflow_name]
        if workflow_type not in self.task_graph_generator.graph_templates:
            return {'valid': False, 'error': f'Workflow type {workflow_type} template not found'}
        test_context = ExecutionContext(
            session_id='validation_test',
            query='Test validation query'
        )
        try:
            tasks = await self.task_graph_generator.generate_dynamic_graph(workflow_type, test_context)
            task_ids = {task.metadata.task_id for task in tasks}
            dependency_errors = []
            for task in tasks:
                for dep in task.metadata.dependencies:
                    if dep not in task_ids:
                        dependency_errors.append(f'Task {task.metadata.task_id} has unresolved dependency: {dep}')
            if dependency_errors:
                return {'valid': False, 'errors': dependency_errors}
            return {
                'valid': True,
                'workflow_type': workflow_type,
                'generated_tasks': len(tasks),
                'task_names': [task.metadata.name for task in tasks]
            }
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}

async def test_dag_engine():
    from .llm import create_llm_client
    from .context_engineer import ContextEngineer, PromptConstructionConfig
    from .vector import HybridVectorStore, VectorStoreConfig
    
    llm = create_llm_client(provider="ollama", model="mistral")
    context_engineer = ContextEngineer(
        PromptConstructionConfig(),
        templates_dir="/Users/mohammednihal/Desktop/ContextChain/ContextChain/contextchain/prompts"
    )
    vector_store = HybridVectorStore(VectorStoreConfig())
    
    engine = DAGEngine(
        max_parallel_tasks=5,
        vector_store=vector_store,
        llm_optimizer=llm,
        context_engineer=context_engineer,
        templates_dir="/Users/mohammednihal/Desktop/ContextChain/ContextChain/contextchain/prompts"
    )
    
    context = ExecutionContext(
        session_id='test_session',
        query='Analyze the Q3 2025 sales performance and identify growth drivers',
        raw_data={'documents': []}
    )
    
    # Validate workflow before execution
    validation = await engine.validate_workflow('historical_analysis')
    if not validation['valid']:
        print(f"Workflow validation failed: {validation['error']}")
        return
    
    result_context = await engine.execute_workflow('historical_analysis', context)
    
    print("DAG Execution Results:")
    print("=" * 50)
    print(f"Workflow Status: {result_context.metadata.get('execution_status')}")
    print(f"Execution Time: {result_context.metadata.get('total_execution_time', 0):.3f}s")
    print(f"Tasks Executed: {result_context.metadata.get('tasks_executed', 0)}")
    print(f"Final Response: {result_context.processed_data.get('final_response', {}).get('generated_response', 'None')}")
    
    stats = engine.get_engine_stats()
    print(f"\nEngine Statistics:")
    print(f"Total Workflows: {stats['engine_stats']['total_workflows']}")
    print(f"Success Rate: {stats['engine_stats']['successful_workflows']}/{stats['engine_stats']['total_workflows']}")
    print(f"Avg Workflow Time: {stats['engine_stats']['avg_workflow_time']:.3f}s")
    print(f"Total Tasks Executed: {stats['engine_stats']['total_tasks_executed']}")
    
    await engine.close()
    return result_context

if __name__ == "__main__":
    asyncio.run(test_dag_engine())