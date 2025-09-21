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
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from .llm import BaseLLMClient, GenerationResult, LLMConfig
from .storage import IntelligentStorage
from .acba import AdaptiveContextBudgetingAlgorithm, BudgetAllocation, QueryComplexity
from .context_engineer import ContextEngineer, PromptStyle, PromptConstructionConfig
from .vector import HybridVectorStore, VectorStoreConfig
from .dag import DAGEngine, ExecutionContext

logger = logging.getLogger(__name__)

class DataDestination(Enum):
    """Possible destinations for data routing"""
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    VECTOR_DB = "vector_db"

class InsightType(Enum):
    """Types of insights generated"""
    FORECAST = "forecast"
    HISTORICAL_STRUCTURED = "historical-structured"
    HISTORICAL_SEMANTIC = "historical-semantic"
    HYBRID = "hybrid"

class ContextChainConfig(BaseModel):
    """Configuration for ContextChain"""
    max_tokens: int = 4096
    default_prompt_style: str = "analytical"
    vector_config: Optional[VectorStoreConfig] = None
    llm_config: Optional[LLMConfig] = None
    prompt_config: Optional[PromptConstructionConfig] = None
    enable_mongo: bool = False
    mongo_uri: str = "mongodb://localhost:27017/contextchain"
    max_parallel_tasks: int = 5
    max_query_length: int = 1000
    max_context_length: int = 50000
    structured_indicators: List[str] = ["forecast", "prediction", "metric", "kpi", "revenue", "sales", "result", "average", "trend"]
    contextual_indicators: List[str] = ["analysis", "insight", "summary", "context", "document"]
    default_destination: DataDestination = DataDestination.MONGODB

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
    destination: Optional[str] = None  # "mongodb" or "postgresql"

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int  # 1-5
    comments: Optional[str] = None

class ContextChain:
    """Main orchestrator for ContextChain v2.0"""
    def __init__(self, config: ContextChainConfig, llm_client: Optional[BaseLLMClient] = None):
        self.config = config
        self.acba = AdaptiveContextBudgetingAlgorithm(max_tokens=config.max_tokens)
        self.context_engineer = ContextEngineer(config.prompt_config or PromptConstructionConfig())
        self.vector_store = HybridVectorStore(config.vector_config or VectorStoreConfig())
        self.dag_engine = DAGEngine(config.max_parallel_tasks)
        self.storage = IntelligentStorage(mongo_uri=config.mongo_uri) if config.enable_mongo else None
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_latency_ms': 0.0,
            'start_time': datetime.utcnow()
        }
        self.structured_query_cache = {}  # Added for caching structured queries
        logger.info("ContextChain v2.0 initialized successfully")

    async def initialize(self):
        """Initialize components"""
        logger.info("Initializing ContextChain components...")
        try:
            if self.storage:
                await self.storage.initialize()
            await self._register_workflows()
            await self._warmup_components()
            logger.info("ContextChain initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContextChain: {str(e)}")
            raise

    async def _register_workflows(self):
        """Register DAG workflows"""
        await self.dag_engine.register_workflow("simple_qa", "simple_retrieval")

    async def _warmup_components(self):
        """Warm up components with health checks"""
        health_check = await self.llm_optimizer.health_check()
        if health_check['status'] != 'healthy':
            logger.warning(f"LLM backend health check failed: {health_check}")

    def _classify_query(self, query: str) -> str:
        """Classify query type for routing (structured, semantic, or hybrid)"""
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in self.config.structured_indicators):
            return "structured"
        elif any(indicator in query_lower for indicator in self.config.contextual_indicators):
            return "semantic"
        else:
            return "hybrid"

    @lru_cache(maxsize=100)  # Added caching for structured queries
    async def _query_structured_db(self, query: str) -> List[Dict[str, Any]]:
        """Fetch structured historical insights from PostgreSQL (with caching)"""
        logger.info(f"Executing structured query: {query}")
        try:
            # Placeholder for SQL query logic (e.g., using asyncpg)
            # In real implementation, replace with actual DB query
            results = []
            if "average" in query.lower():
                results = [{"metric": "average_sales", "value": 12345.67, "period": "2025-Q3"}]
            elif "trend" in query.lower():
                results = [{"metric": "sales_trend", "value": 5.2, "unit": "percent", "period": "2024-2025"}]
            return [
                {
                    "content": str(result),
                    "source_id": f"postgres_{uuid.uuid4()}",
                    "metadata": {"type": "structured", "metric": result.get("metric")},
                    "score": 1.0,
                    "embedding_quality": 1.0
                } for result in results
            ]
        except Exception as e:
            logger.warning(f"Structured query failed: {str(e)}, returning empty results")
            return []

    async def generate_insights(self, query: str, context: Optional[str] = None,
                               documents: Optional[List[Dict[str, Any]]] = None,
                               session_id: Optional[str] = None,
                               mode: str = "auto", **kwargs) -> ContextChainResponse:
        """Generate insights with intelligent processing"""
        start_time = time.time()
        processing_steps = []
        session_id = session_id or str(uuid.uuid4())
        data_routing = {}
        
        try:
            await self._validate_input(query, documents)
            processing_steps.append({
                'step': 'input_validation',
                'status': 'completed',
                'timestamp': datetime.utcnow()
            })
            
            execution_context = ExecutionContext(session_id=session_id, query=query, raw_data={'documents': documents or []})
            
            complexity = self.acba.assess_complexity(query)
            processing_steps.append({
                'step': 'complexity_assessment',
                'status': 'completed',
                'complexity_score': complexity.overall_score,
                'timestamp': datetime.utcnow()
            })
            
            if mode == "auto":
                insight_type = self._classify_query(query)
            else:
                if mode not in ["vector", "structured", "hybrid"]:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'vector', 'structured', 'hybrid', or 'auto'.")
                insight_type = "historical-semantic" if mode == "vector" else mode

            if documents:
                retrieved_docs = documents
            elif mode == "structured":
                retrieved_docs = await self._query_structured_db(query)
                insight_type = InsightType.HISTORICAL_STRUCTURED.value
            elif mode == "vector":
                retrieved_docs = await self._retrieve_documents(query, execution_context)
                insight_type = InsightType.HISTORICAL_SEMANTIC.value
            else:  # hybrid or auto
                docs_structured = await self._query_structured_db(query)
                docs_semantic = await self._retrieve_documents(query, execution_context)
                retrieved_docs = docs_structured + docs_semantic
                insight_type = InsightType.HYBRID.value if insight_type == "hybrid" else InsightType[insight_type.upper()].value
            
            execution_context.raw_data['documents'] = retrieved_docs
            processing_steps.append({
                'step': 'document_retrieval',
                'status': 'completed',
                'documents_count': len(retrieved_docs),
                'insight_type': insight_type,
                'timestamp': datetime.utcnow()
            })
            
            budget = await self.acba.compute_optimal_budget(query, retrieved_docs, context={'session_id': session_id})
            execution_context.budget_allocation = budget
            processing_steps.append({
                'step': 'budget_allocation',
                'status': 'completed',
                'arm_selected': budget.arm_selected,
                'total_budget': budget.total_budget,
                'timestamp': datetime.utcnow()
            })
            
            workflow_map = {
                'structured': 'historical_analysis',
                'semantic': 'simple_qa',
                'hybrid': 'complex_analytical'
            }
            workflow_name = workflow_map.get(insight_type, 'simple_qa')
            final_context = await self.dag_engine.execute_workflow(workflow_name, execution_context)
            processing_steps.append({
                'step': 'dag_execution',
                'status': 'completed',
                'workflow_used': workflow_name,
                'timestamp': datetime.utcnow()
            })
            
            optimized_context = await self.context_engineer.build_prompt(
                query=query,
                raw_docs=retrieved_docs,
                budget=budget,
                complexity=complexity,
                prompt_style=PromptStyle(self.config.default_prompt_style),
                semantic_state=final_context.semantic_state
            )
            processing_steps.append({
                'step': 'context_engineering',
                'status': 'completed',
                'timestamp': datetime.utcnow()
            })
            
            llm_response = await self.llm_optimizer.generate_optimized(
                prompt=optimized_context,
                budget=budget,
                stream=kwargs.get('stream', False)
            )
            processing_steps.append({
                'step': 'llm_generation',
                'status': 'completed',
                'tokens_used': llm_response.tokens_used,
                'timestamp': datetime.utcnow()
            })
            
            total_latency_ms = (time.time() - start_time) * 1000
            quality_score = self._assess_response_quality(llm_response, complexity)
            tokens_saved = self._calculate_tokens_saved(retrieved_docs, budget)
            
            if self.storage:
                data_routing = await self.store_business_data(
                    data_type="insight_response",
                    data=llm_response.content,
                    metadata={'session_id': session_id, 'query': query[:100], 'insight_type': insight_type},
                    destination=self.config.default_destination.value
                )
            
            response = ContextChainResponse(
                response=llm_response.content,
                session_id=session_id,
                total_latency_ms=total_latency_ms,
                tokens_used=llm_response.tokens_used,
                tokens_saved=tokens_saved,
                quality_score=quality_score,
                budget_allocation=budget.__dict__,
                data_routing=data_routing,
                processing_steps=processing_steps,
                documents_used=len(retrieved_docs),
                timestamp=datetime.utcnow(),
                success=True,
                insight_type=insight_type
            )
            
            if self.storage:
                asyncio.create_task(self._log_interaction(query, complexity, budget, response, final_context))
            
            self._update_performance_stats(response)
            return response
            
        except Exception as e:
            return await self._handle_error(e, query, session_id, processing_steps, start_time)

    async def _validate_input(self, query: str, documents: Optional[List[Dict]] = None):
        """Validate input query and documents"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > self.config.max_query_length:
            raise ValueError(f"Query too long: {len(query)} > {self.config.max_query_length}")
        if documents and sum(len(str(doc)) for doc in documents) > self.config.max_context_length:
            raise ValueError(f"Context too long")

    async def _retrieve_documents(self, query: str, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector store"""
        try:
            search_results = await self.vector_store.search(query=query, k=5)
            return [
                {
                    'content': result.content,
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
        """Assess response quality based on length and complexity"""
        response_length = len(llm_response.content.split())
        target_length = complexity.overall_score * 100
        length_score = min(target_length / max(response_length, 1), 1.0)
        return length_score

    def _calculate_tokens_saved(self, documents: List[Dict], budget: BudgetAllocation) -> int:
        """Calculate tokens saved during optimization"""
        total_possible_tokens = sum(len(str(doc)) for doc in documents) // 4  # Rough token estimation
        return max(0, total_possible_tokens - budget.generation_tokens)

    async def _log_interaction(self, query: str, complexity: QueryComplexity, budget: BudgetAllocation,
                              response: ContextChainResponse, context: ExecutionContext):
        """Log interaction to storage"""
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
                complexity=complexity.dict(),
                budget_allocation=budget.dict(),
                performance_metrics=performance_metrics,
                success=response.success,
                error_message=response.error_message
            )
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")

    def _update_performance_stats(self, response: ContextChainResponse):
        """Update performance statistics"""
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
        """Handle errors during processing"""
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
        """Index documents in vector store"""
        try:
            doc_ids = await self.vector_store.index_documents(documents)
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
        """Get system status and component health"""
        try:
            llm_health = await self.llm_optimizer.health_check()
            vector_stats = self.vector_store.get_performance_stats()
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
        """Intelligent routing decision for metadata storage (fallback)"""
        content_lower = content.lower()
        if data_type in self.config.structured_indicators or any(indicator in content_lower for indicator in self.config.structured_indicators):
            return DataDestination.POSTGRESQL
        elif data_type in self.config.contextual_indicators or any(indicator in content_lower for indicator in self.config.contextual_indicators):
            return DataDestination.VECTOR_DB
        else:
            return self.config.default_destination

    async def store_business_data(self, data_type: str, data: Any, metadata: Optional[Dict] = None,
                                 destination: Optional[str] = None) -> Dict[str, str]:
        """Store business data with user-specified destination or fallback routing"""
        import json
        content = json.dumps(data) if not isinstance(data, str) else data
        routing = {}
        
        # Use user-specified destination for MongoDB or PostgreSQL
        if destination:
            if destination not in ["mongodb", "postgresql"]:
                raise ValueError(f"Invalid destination: {destination}. Must be 'mongodb' or 'postgresql'.")
            if destination == "mongodb" and not self.storage:
                raise ValueError("MongoDB storage not enabled. Set enable_mongo=True in config.")
            if destination == "mongodb":
                doc_id = await self.storage._store_to_mongodb(data_type, content, metadata or {})
                routing["mongodb"] = doc_id
            elif destination == "postgresql":
                routing["postgresql"] = "handled_by_application"
        else:
            # Fallback to intelligent routing for metadata
            dest = self.decide_data_destination(data_type, content, metadata)
            if dest == DataDestination.MONGODB and self.storage:
                doc_id = await self.storage._store_to_mongodb(data_type, content, metadata or {})
                routing["mongodb"] = doc_id
            elif dest == DataDestination.POSTGRESQL:
                routing["postgresql"] = "handled_by_application"
            elif dest == DataDestination.VECTOR_DB:
                doc_id = await self.vector_store.index_documents([{"content": content, "metadata": metadata or {}}])
                routing["vector_db"] = doc_id[0] if doc_id else "none"
        
        # Store metadata in MongoDB if enabled
        if self.storage:
            meta_id = await self.storage._store_metadata(data_type, content, metadata or {}, destination or dest.value)
            routing["metadata"] = meta_id
        
        return routing

    async def search_context(self, query: str, k: int = 5, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search stored context in vector store"""
        return await self.vector_store.search(query=query, k=k, metadata_filter={"data_type": data_type} if data_type else None)

    async def submit_feedback(self, session_id: str, rating: int, comments: Optional[str] = None) -> Dict[str, str]:
        """Store feedback for a session"""
        try:
            if not self.storage:
                raise ValueError("MongoDB storage not enabled for feedback")
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5")
            feedback_id = await self.storage.store_feedback(session_id, rating, comments)
            return {"status": "success", "feedback_id": feedback_id}
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close resources"""
        if self.storage:
            self.storage.close()
        if hasattr(self.llm_optimizer, 'close'):
            await self.llm_optimizer.close()
        await self.context_engineer.close()
        await self.dag_engine.close()
        await self.vector_store.close()

def create_contextchain_app(config: ContextChainConfig = None, llm_client: Optional[BaseLLMClient] = None) -> FastAPI:
    """Library entry point to create a FastAPI application with ContextChain integration"""
    config = config or ContextChainConfig()
    context_chain = ContextChain(config, llm_client=llm_client)
    
    app = FastAPI(
        title="ContextChain v2.0 API",
        description="Advanced context optimization for LLM applications",
        version="2.0.0"
    )
    
    # Rate limiting middleware
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"➡️ {request.method} {request.url}")
        start_time = time.time()
        response = await call_next(request)
        logger.info(f"⬅️ {response.status_code} for {request.url} in {(time.time() - start_time) * 1000:.2f}ms")
        return response
    
    # Lifespan events
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"ContextChain config: {config.dict()}")
        await context_chain.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await context_chain.close()
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return await context_chain.get_system_status()
    
    # Metrics endpoint with Prometheus
    @app.get("/metrics")
    async def get_metrics():
        try:
            return {
                "status": "success",
                "metrics": context_chain.performance_stats,
                "vector_store_stats": context_chain.vector_store.get_performance_stats(),
                "llm_stats": await context_chain.llm_optimizer.get_performance_stats()
            }
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Query endpoint with rate limiting
    @app.post("/query")
    @app.state.limiter.limit("10/minute")  # Rate limit example: 10 requests/min per IP
    async def query_insights(request: Request, query: str, context: Optional[str] = None,
                             documents: Optional[List[Dict[str, Any]]] = None,
                             session_id: Optional[str] = None,
                             mode: str = "auto",
                             stream: bool = False):
        try:
            response = await context_chain.generate_insights(
                query=query,
                context=context,
                documents=documents,
                session_id=session_id,
                mode=mode,
                stream=stream
            )
            if stream:
                async def stream_generator():
                    yield json.dumps(response.dict())  # Stream as JSON chunks if needed
                return StreamingResponse(stream_generator(), media_type="application/json")
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Store data endpoint
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
    
    # Index documents endpoint (for Vector DB)
    @app.post("/index")
    async def index_documents(documents: List[Dict[str, Any]]):
        try:
            result = await context_chain.index_documents(documents)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Search context endpoint
    @app.get("/search")
    async def search_context(query: str, k: int = 5, data_type: Optional[str] = None):
        try:
            results = await context_chain.search_context(query=query, k=k, data_type=data_type)
            return {"status": "success", "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Feedback endpoint
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
    
    # Instrumentator for Prometheus metrics
    Instrumentator().instrument(app).expose(app)
    
    return app

if __name__ == "__main__":
    import uvicorn
    app = create_contextchain_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)