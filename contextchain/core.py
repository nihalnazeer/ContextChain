"""
Core orchestrator for ContextChain v2.0
Manages LLM, vector store, storage, and intelligent data routing
Provides library entry point for FastAPI integration with full endpoint support
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import asyncio
import logging
import time
import json
import uuid
import re
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from .llm import BaseLLMClient, GenerationResult, LLMConfig, create_llm_client
from .storage import IntelligentStorage
from .acba import AdaptiveContextBudgetingAlgorithm, BudgetAllocation, QueryComplexity
from .context_engineer import ContextEngineer, PromptStyle, PromptConstructionConfig
from .vector import HybridVectorStore, VectorStoreConfig
from .dag import DAGEngine, ExecutionContext
import os

logger = logging.getLogger(__name__)

class DataDestination(Enum):
    """Possible destinations for data routing"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"  # Repurposed from MONGODB
    VECTOR_DB = "vector_db"

class InsightType(Enum):
    """Types of insights generated"""
    FORECAST = "forecast"
    HISTORICAL_STRUCTURED = "historical-structured"
    HISTORICAL_SEMANTIC = "historical-semantic"
    HYBRID = "hybrid"
    FILE = "file"  # New for file source

class DataPreprocessor:
    """Preprocesses incoming data by cleaning, normalizing, and enriching text"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.remove_special_chars = self.config.get('remove_special_chars', True)
        self.lowercase = self.config.get('lowercase', True)
        self.remove_extra_spaces = self.config.get('remove_extra_spaces', True)
        self.special_char_pattern = re.compile(r'[^\w\s.,!?]')  # Keep alphanumeric, spaces, and basic punctuation
    
    def preprocess(self, text: str) -> str:
        """Apply preprocessing transformations to the input text"""
        if not text or not isinstance(text, str):
            return text
        
        try:
            processed = text
            if self.lowercase:
                processed = processed.lower()
            if self.remove_special_chars:
                processed = self.special_char_pattern.sub('', processed)
            if self.remove_extra_spaces:
                processed = ' '.join(processed.split())
            return processed.strip()
        except Exception as e:
            logger.warning(f"Preprocessing failed for text: {text[:50]}... Error: {str(e)}")
            return text

class ContextChainConfig(BaseModel):
    """Configuration for ContextChain"""
    max_tokens: int = 4096
    default_prompt_style: str = "analytical"
    vector_config: Optional[VectorStoreConfig] = None
    llm_config: Optional[LLMConfig] = None
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    prompt_config: Optional[PromptConstructionConfig] = None
    prompt_quality_threshold: float = float(os.getenv("PROMPT_QUALITY_THRESHOLD", "0.5"))
    db_path: str = "contextchain.db"
    max_parallel_tasks: int = 5
    max_query_length: int = 1000
    max_context_length: int = 50000
    structured_indicators: List[str] = ["forecast", "prediction", "metric", "kpi", "revenue", "sales", "result", "average", "trend"]
    contextual_indicators: List[str] = ["analysis", "insight", "summary", "context", "document"]
    default_destination: DataDestination = DataDestination.SQLITE
    preprocessor_config: Optional[Dict[str, Any]] = None
    enable_vector: bool = True  # New: Toggle vector store
    file_paths: Optional[List[str]] = None  # Optional default files

    class Config:
        arbitrary_types_allowed = True

class ContextChainResponse(BaseModel):
    response: str
    session_id: str
    total_latency_ms: float
    tokens_used: int
    tokens_saved: int
    quality_score: float
    budget_allocation: Dict[str, Any]
    data_routing: Dict[str, str]
    processing_steps: List[Dict[str, Any]]
    documents_used: int
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    insight_type: str

class StoreDataRequest(BaseModel):
    data_type: str
    data: Any
    metadata: Optional[Dict] = None
    destination: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int
    comments: Optional[str] = None

class ContextChain:
    """Main orchestrator for ContextChain v2.0"""
    def __init__(self, config: ContextChainConfig, llm_client: Optional[BaseLLMClient] = None):
        self.config = config
        self.llm_optimizer = llm_client or create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=config.llm_api_key,
            api_base=config.llm_config.api_base if config.llm_config else None,
            max_tokens=config.llm_config.max_tokens if config.llm_config else 2048,
            temperature=config.llm_config.temperature if config.llm_config else 0.7,
            timeout=config.llm_config.timeout if config.llm_config else 30,
            enable_streaming=config.llm_config.enable_streaming if config.llm_config else False,
            enable_caching=config.llm_config.enable_caching if config.llm_config else True,
            retry_attempts=config.llm_config.retry_attempts if config.llm_config else 3,
            retry_delay=config.llm_config.retry_delay if config.llm_config else 1.0,
            device=config.llm_config.device if config.llm_config else "cpu"
        )
        self.acba = AdaptiveContextBudgetingAlgorithm(max_tokens=config.max_tokens)
        prompt_cfg = config.prompt_config or PromptConstructionConfig(quality_threshold=config.prompt_quality_threshold)
        self.context_engineer = ContextEngineer(prompt_cfg)
        self.vector_store = HybridVectorStore(config.vector_config or VectorStoreConfig()) if config.enable_vector else None
        # FIXED: Properly inject shared components into DAGEngine for dependency resolution
        # This ensures tasks like BudgetAllocationTask receive required instances (e.g., acba)
        self.dag_engine = DAGEngine(
            max_parallel_tasks=config.max_parallel_tasks,
            vector_store=self.vector_store,
            llm_optimizer=self.llm_optimizer,
            context_engineer=self.context_engineer,
            acba=self.acba
        )
        self.storage = IntelligentStorage(db_path=config.db_path)
        self.preprocessor = DataPreprocessor(config.preprocessor_config or {})
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_latency_ms': 0.0,
            'start_time': datetime.utcnow()
        }
        logger.info(f"ContextChain v2.0 initialized with LLM provider: {config.llm_provider}, model: {config.llm_model}, quality_threshold: {prompt_cfg.quality_threshold}, vector_enabled: {config.enable_vector}")

    async def initialize(self):
        logger.info("Initializing ContextChain components...")
        try:
            await self.storage.initialize()
            await self._register_workflows()
            await self._warmup_components()
            logger.info("ContextChain initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContextChain: {str(e)}")
            raise

    async def _register_workflows(self):
        await self.dag_engine.register_workflow("simple_qa", "simple_retrieval")

    async def _warmup_components(self):
        health_check = await self.llm_optimizer.health_check()
        if health_check['status'] != 'healthy':
            logger.warning(f"LLM backend health check failed: {health_check}")

    def _classify_query(self, query: str) -> str:
        query_lower = self.preprocessor.preprocess(query).lower()
        if any(indicator in query_lower for indicator in self.config.structured_indicators):
            return "relational"
        elif any(indicator in query_lower for indicator in self.config.contextual_indicators):
            return "vector"
        else:
            return "hybrid"

    @lru_cache(maxsize=100)
    async def _query_structured_db(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Executing structured query: {query}")
        try:
            results = []
            if "average" in query.lower():
                results = [{"metric": "average_sales", "value": 12345.67, "period": "2025-Q3"}]
            elif "trend" in query.lower():
                results = [{"metric": "sales_trend", "value": 5.2, "unit": "percent", "period": "2024-2025"}]
            processed_results = [
                {
                    "content": self.preprocessor.preprocess(str(result)),
                    "source_id": f"postgres_{uuid.uuid4()}",
                    "metadata": {"type": "relational", "metric": result.get("metric")},
                    "score": 1.0,
                    "embedding_quality": 1.0
                } for result in results
            ]
            return processed_results
        except Exception as e:
            logger.warning(f"Structured query failed: {str(e)}, returning empty results")
            return []

    async def _retrieve_from_file(self, file_path: str, query: str) -> List[Dict[str, Any]]:
        import pandas as pd
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file type")
            mask = df.apply(lambda row: query.lower() in ' '.join(row.astype(str)).lower(), axis=1)
            filtered = df[mask].to_dict(orient='records')
            return [
                {
                    'content': str(record),
                    'source_id': f"file_{uuid.uuid4()}",
                    'metadata': {'type': 'file', 'path': file_path},
                    'score': 1.0,
                    'embedding_quality': 1.0
                } for record in filtered
            ]
        except Exception as e:
            logger.warning(f"File retrieval failed: {str(e)}, returning empty results")
            return []

    async def generate_insights(self, query: str,
                               documents: Optional[List[Dict[str, Any]]] = None,
                               session_id: Optional[str] = None,
                               mode: str = "auto",
                               data_source: str = "auto",
                               file_path: Optional[str] = None,
                               **kwargs) -> ContextChainResponse:
        start_time = time.time()
        processing_steps = []
        session_id = session_id or str(uuid.uuid4())
        data_routing = {}
        
        try:
            preprocessed_query = self.preprocessor.preprocess(query)
            await self._validate_input(preprocessed_query, documents)
            processing_steps.append({'step': 'input_validation', 'status': 'completed', 'timestamp': datetime.utcnow()})
            
            preprocessed_docs = None
            if documents:
                preprocessed_docs = [{**doc, "content": self.preprocessor.preprocess(doc["content"])} for doc in documents]
            
            execution_context = ExecutionContext(session_id=session_id, query=preprocessed_query, raw_data={'documents': preprocessed_docs or []})
            
            complexity = self.acba.complexity_assessor.assess_complexity(preprocessed_query)
            processing_steps.append({'step': 'complexity_assessment', 'status': 'completed', 'complexity_score': complexity.overall_score, 'timestamp': datetime.utcnow()})
            
            if data_source != "auto":
                insight_type = data_source
            else:
                insight_type = self._classify_query(preprocessed_query)
            
            if preprocessed_docs:
                retrieved_docs = preprocessed_docs
                insight_type_enum = InsightType.HISTORICAL_SEMANTIC.value if "vector" in insight_type else insight_type
            elif insight_type == "relational":
                retrieved_docs = await self._query_structured_db(preprocessed_query)
                insight_type_enum = InsightType.HISTORICAL_STRUCTURED.value
            elif insight_type == "vector":
                if not self.vector_store:
                    raise ValueError("Vector store is disabled")
                retrieved_docs = await self._retrieve_documents(preprocessed_query, execution_context)
                insight_type_enum = InsightType.HISTORICAL_SEMANTIC.value
            elif insight_type == "file":
                if not file_path:
                    raise ValueError("file_path required for data_source='file'")
                retrieved_docs = await self._retrieve_from_file(file_path, preprocessed_query)
                insight_type_enum = InsightType.FILE.value
            else:
                docs_relational = await self._query_structured_db(preprocessed_query)
                docs_vector = await self._retrieve_documents(preprocessed_query, execution_context) if self.vector_store else []
                retrieved_docs = docs_relational + docs_vector
                insight_type_enum = InsightType.HYBRID.value
            
            if any(not doc.get('content_preprocessed') for doc in retrieved_docs if 'content_preprocessed' in doc):
                retrieved_docs = [{**doc, 'content': self.preprocessor.preprocess(doc['content'])} for doc in retrieved_docs]
            
            execution_context.raw_data['documents'] = retrieved_docs
            processing_steps.append({'step': 'document_retrieval', 'status': 'completed', 'documents_count': len(retrieved_docs), 'insight_type': insight_type_enum, 'timestamp': datetime.utcnow()})
            
            budget = await self.acba.compute_optimal_budget(preprocessed_query, retrieved_docs, context={'session_id': session_id})
            execution_context.budget_allocation = budget
            processing_steps.append({'step': 'budget_allocation', 'status': 'completed', 'arm_selected': budget.arm_selected, 'total_budget': budget.total_budget, 'timestamp': datetime.utcnow()})
            
            workflow_map = {
                'relational': 'historical_analysis',
                'vector': 'simple_qa',
                'hybrid': 'complex_analytical',
                'file': 'simple_qa'  # Reuse for file
            }
            workflow_name = workflow_map.get(insight_type, 'simple_qa')
            final_context = await self.dag_engine.execute_workflow(workflow_name, execution_context)
            processing_steps.append({'step': 'dag_execution', 'status': 'completed', 'workflow_used': workflow_name, 'timestamp': datetime.utcnow()})
            
            optimized_context = await self.context_engineer.build_prompt(
                query=preprocessed_query,
                raw_docs=retrieved_docs,
                budget=budget,
                complexity=complexity
            )
            processing_steps.append({'step': 'context_engineering', 'status': 'completed', 'timestamp': datetime.utcnow()})
            
            llm_response = await self.llm_optimizer.generate_optimized(
                prompt=optimized_context,
                budget=budget,
                stream=kwargs.get('stream', False)
            )
            processing_steps.append({'step': 'llm_generation', 'status': 'completed', 'tokens_used': llm_response.tokens_used, 'timestamp': datetime.utcnow()})
            
            total_latency_ms = (time.time() - start_time) * 1000
            quality_score = self._assess_response_quality(llm_response, complexity)
            tokens_saved = self._calculate_tokens_saved(retrieved_docs, budget)
            
            data_routing = await self.store_business_data(
                data_type="insight_response",
                data=llm_response.content,
                metadata={'session_id': session_id, 'query': preprocessed_query[:100], 'insight_type': insight_type_enum},
                destination=self.config.default_destination.value
            )
            
            response = ContextChainResponse(
                response=llm_response.content,
                session_id=session_id,
                total_latency_ms=total_latency_ms,
                tokens_used=llm_response.tokens_used,
                tokens_saved=tokens_saved,
                quality_score=quality_score,
                budget_allocation=vars(budget),
                data_routing=data_routing,
                processing_steps=processing_steps,
                documents_used=len(retrieved_docs),
                timestamp=datetime.utcnow(),
                success=True,
                insight_type=insight_type_enum
            )
            
            asyncio.create_task(self._log_interaction(preprocessed_query, complexity, budget, response, final_context))
            
            self._update_performance_stats(response)
            return response
            
        except Exception as e:
            return await self._handle_error(e, preprocessed_query or query, session_id, processing_steps, start_time)

    async def _validate_input(self, query: str, documents: Optional[List[Dict]] = None):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > self.config.max_query_length:
            raise ValueError(f"Query too long: {len(query)} > {self.config.max_query_length}")
        if documents and sum(len(str(doc)) for doc in documents) > self.config.max_context_length:
            raise ValueError(f"Context too long")

    async def _retrieve_documents(self, query: str, context: ExecutionContext) -> List[Dict[str, Any]]:
        try:
            search_results = await self.vector_store.search(query=query, k=5)
            return [
                {
                    'content': self.preprocessor.preprocess(result.content),
                    'source_id': result.source_id,
                    'score': result.fusion_score,
                    'metadata': result.metadata,
                    'embedding_quality': result.embedding_quality
                } for result in search_results
            ]
        except Exception as e:
            logger.warning(f"Document retrieval failed: {str(e)}, using empty context")
            return []

    def _assess_response_quality(self, llm_response: GenerationResult, complexity: QueryComplexity) -> float:
        response_length = len(llm_response.content.split())
        target_length = complexity.overall_score * 100
        length_score = min(target_length / max(response_length, 1), 1.0)
        return length_score

    def _calculate_tokens_saved(self, documents: List[Dict], budget: BudgetAllocation) -> int:
        total_possible_tokens = sum(len(str(doc)) for doc in documents) // 4
        return max(0, total_possible_tokens - budget.generation_tokens)

    async def _log_interaction(self, query: str, complexity: QueryComplexity, budget: BudgetAllocation,
                              response: ContextChainResponse, context: ExecutionContext):
        try:
            performance_metrics = {
                'documents_retrieved': response.documents_used,
                'total_latency_ms': response.total_latency_ms,
                'tokens_generated': response.tokens_used,
                'tokens_saved': response.tokens_saved,
                'insight_type': response.insight_type
            }
            await self.storage.log_interaction(
                session_id=response.session_id,
                query=query,
                complexity=vars(complexity),
                budget_allocation=vars(budget),
                performance_metrics=performance_metrics,
                success=response.success,
                error_message=response.error_message
            )
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")

    def _update_performance_stats(self, response: ContextChainResponse):
        self.performance_stats['total_requests'] += 1
        if response.success:
            self.performance_stats['successful_requests'] += 1
        total_requests = self.performance_stats['total_requests']
        current_avg = self.performance_stats['avg_latency_ms']
        self.performance_stats['avg_latency_ms'] = (
            (current_avg * (total_requests - 1) + response.total_latency_ms) / total_requests
        )

    async def _handle_error(self, error: Exception, query: str, session_id: str,
                           processing_steps: List[Dict], start_time: float) -> ContextChainResponse:
        logger.error(f"Error processing query '{query[:100]}...': {str(error)}")
        processing_steps.append({
            'step': 'error_handling',
            'status': 'error',
            'error': str(error),
            'timestamp': datetime.utcnow()
        })
        total_latency_ms = (time.time() - start_time) * 1000
        error_response = ContextChainResponse(
            response=f"Error: {str(error)[:200]}",
            session_id=session_id,
            total_latency_ms=total_latency_ms,
            tokens_used=0,
            tokens_saved=0,
            quality_score=0.0,
            budget_allocation={},
            data_routing={},
            processing_steps=processing_steps,
            documents_used=0,
            timestamp=datetime.utcnow(),
            success=False,
            error_message=str(error),
            insight_type="none"
        )
        self._update_performance_stats(error_response)
        return error_response

    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.vector_store:
            return {'status': 'error', 'error': 'Vector store is disabled'}
        try:
            processed_docs = [
                {**doc, "content": self.preprocessor.preprocess(doc["content"])}
                for doc in documents
            ]
            doc_ids = await self.vector_store.index_documents(processed_docs)
            stats = self.vector_store.get_performance_stats()
            return {
                'status': 'success',
                'documents_indexed': len(documents),
                'document_ids': doc_ids,
                'total_documents': stats['total_documents']
            }
        except Exception as e:
            logger.error(f"Document indexing failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        try:
            llm_health = await self.llm_optimizer.health_check()
            vector_stats = self.vector_store.get_performance_stats() if self.vector_store else {"status": "disabled"}
            return {
                'status': 'healthy',
                'uptime_seconds': (datetime.utcnow() - self.performance_stats['start_time']).total_seconds(),
                'performance': self.performance_stats,
                'components': {
                    'llm_backend': llm_health,
                    'vector_store': vector_stats
                }
            }
        except Exception as e:
            logger.error(f"System status check failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def decide_data_destination(self, data_type: str, content: str, metadata: Optional[Dict] = None) -> DataDestination:
        content_lower = content.lower()
        if data_type in self.config.structured_indicators or any(indicator in content_lower for indicator in self.config.structured_indicators):
            return DataDestination.POSTGRESQL
        elif data_type in self.config.contextual_indicators or any(indicator in content_lower for indicator in self.config.contextual_indicators):
            return DataDestination.VECTOR_DB
        else:
            return self.config.default_destination

    async def store_business_data(self, data_type: str, data: Any, metadata: Optional[Dict] = None,
                                 destination: Optional[str] = None) -> Dict[str, str]:
        content = json.dumps(data) if not isinstance(data, str) else data
        content = self.preprocessor.preprocess(content)
        routing = {}
        
        if destination:
            if destination not in ["sqlite", "postgresql"]:
                raise ValueError(f"Invalid destination: {destination}. Must be 'sqlite' or 'postgresql'.")
            if destination == "sqlite":
                doc_id = await self.storage._store_to_sqlite(data_type, content, metadata or {})
                routing["sqlite"] = doc_id
            elif destination == "postgresql":
                routing["postgresql"] = "handled_by_application"
        else:
            dest = self.decide_data_destination(data_type, content, metadata)
            if dest == DataDestination.SQLITE:
                doc_id = await self.storage._store_to_sqlite(data_type, content, metadata or {})
                routing["sqlite"] = doc_id
            elif dest == DataDestination.POSTGRESQL:
                routing["postgresql"] = "handled_by_application"
            elif dest == DataDestination.VECTOR_DB:
                if not self.vector_store:
                    routing["vector_db"] = "disabled"
                else:
                    doc_id = await self.vector_store.index_documents([{"content": content, "metadata": metadata or {}}])
                    routing["vector_db"] = doc_id[0] if doc_id else "none"
        
        meta_id = await self.storage._store_metadata(data_type, content, metadata or {}, destination or dest.value)
        routing["metadata"] = meta_id
        
        return routing

    async def search_context(self, query: str, k: int = 5, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("Vector store is disabled")
        query = self.preprocessor.preprocess(query)
        return await self.vector_store.search(query=query, k=k, metadata_filter={"data_type": data_type} if data_type else None)

    async def submit_feedback(self, session_id: str, rating: int, comments: Optional[str] = None) -> Dict[str, str]:
        try:
            if comments:
                comments = self.preprocessor.preprocess(comments)
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5")
            feedback_id = await self.storage.store_feedback(session_id, rating, comments)
            return {"status": "success", "feedback_id": feedback_id}
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        await self.storage.close()
        if hasattr(self.llm_optimizer, 'close'):
            await self.llm_optimizer.close()
        if hasattr(self.context_engineer, 'close'):
            await self.context_engineer.close() 
        await self.dag_engine.close()
        if self.vector_store:
            await self.vector_store.close()

def create_contextchain_app(config: ContextChainConfig = None, llm_client: Optional[BaseLLMClient] = None) -> FastAPI:
    config = config or ContextChainConfig()
    context_chain = ContextChain(config, llm_client=llm_client)
    
    app = FastAPI(
        title="ContextChain v2.0 API",
        description="Advanced context optimization for LLM applications",
        version="2.0.0"
    )
    
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"➡️ {request.method} {request.url}")
        start_time = time.time()
        response = await call_next(request)
        logger.info(f"⬅️ {response.status_code} for {request.url} in {(time.time() - start_time) * 1000:.2f}ms")
        return response
    
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"ContextChain config: {config.dict()}")
        await context_chain.initialize()
        app.state.contextchain = context_chain
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await context_chain.close()
    
    @app.get("/health")
    async def health_check():
        return await context_chain.get_system_status()
    
    @app.get("/metrics")
    async def get_metrics():
        try:
            stats = {
                "performance": context_chain.performance_stats,
                "vector_store": context_chain.vector_store.get_performance_stats() if context_chain.vector_store else {"status": "disabled"},
                "llm": await context_chain.llm_optimizer.get_performance_stats()
            }
            return {"status": "success", "metrics": stats}
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/query")
    @app.state.limiter.limit("10/minute")
    async def query_insights(request: Request, query: str,
                             documents: Optional[List[Dict[str, Any]]] = None,
                             session_id: Optional[str] = None,
                             mode: str = "auto",
                             data_source: str = "auto",  # New
                             file_path: Optional[str] = None,  # New
                             stream: bool = False):
        try:
            response = await context_chain.generate_insights(
                query=query,
                documents=documents,
                session_id=session_id,
                mode=mode,
                data_source=data_source,
                file_path=file_path,
                stream=stream
            )
            if stream:
                async def stream_generator():
                    yield json.dumps(response.dict())
                return StreamingResponse(stream_generator(), media_type="application/json")
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/store")
    async def store_data(request: StoreDataRequest):
        try:
            routing = await context_chain.store_business_data(
                data_type=request.data_type,
                data=request.data,
                metadata=request.metadata,
                destination=request.destination
            )
            return {"status": "success", "routing": routing}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/index")
    async def index_documents(documents: List[Dict[str, Any]]):
        try:
            result = await context_chain.index_documents(documents)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/search")
    async def search_context(query: str, k: int = 5, data_type: Optional[str] = None):
        try:
            results = await context_chain.search_context(query=query, k=k, data_type=data_type)
            return {"status": "success", "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/feedback")
    async def submit_feedback(request: FeedbackRequest):
        try:
            result = await context_chain.submit_feedback(
                session_id=request.session_id,
                rating=request.rating,
                comments=request.comments
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    app.context_chain = context_chain
    Instrumentator().instrument(app).expose(app)
    
    return app

if __name__ == "__main__":
    import uvicorn
    app = create_contextchain_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)