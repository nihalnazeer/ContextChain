# contextchain/src/vector.py
"""
Advanced Vector Store with Multi-Vector Architecture
Addressing: Vector Bottleneck (2025), Embedding-Informed Adaptive Retrieval (Huang et al.)
- Multi-vector dense and sparse retrieval
- Embedding quality assessment and fusion
- Persistent storage support via ChromaDB or FAISS
- Async initialization and usage compatible with ContextChain core.py
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle

# Vector storage and similarity
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector search with comprehensive metadata"""
    content: str
    vector_score: float
    sparse_score: float
    fusion_score: float
    source_id: str
    metadata: Dict[str, Any]
    embedding_quality: float
    multi_vector_scores: Dict[str, float]

@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations"""
    dense_model_name: str = "all-MiniLM-L6-v2"
    sparse_model_features: int = 10000
    collection_name: str = "contextchain_docs"
    persist_directory: str = "./chroma_db"
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    fusion_weights: Dict[str, float] = None
    enable_reranking: bool = True
    quality_threshold: float = 0.65
    max_results: int = 100

class MultiVectorEncoder:
    """
    Multi-vector encoding to address compositional query limitations
    Based on Vector Bottleneck research (2025)
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.dense_encoder = self._initialize_dense_encoder()
        self.sparse_encoder = self._initialize_sparse_encoder()
        
        # Multi-vector components for compositional queries
        self.aspect_encoders = self._initialize_aspect_encoders()
        
    def _initialize_dense_encoder(self):
        """Initialize dense sentence transformer encoder"""
        if SentenceTransformer is None:
            logger.warning("SentenceTransformer not available, using fallback")
            return None
            
        try:
            model = SentenceTransformer(self.config.dense_model_name)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error initializing dense encoder: {e}")
            return None
    
    def _initialize_sparse_encoder(self):
        """Initialize sparse TF-IDF encoder"""
        return TfidfVectorizer(
            max_features=self.config.sparse_model_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    
    def _initialize_aspect_encoders(self):
        """Initialize aspect-specific encoders for multi-vector approach"""
        aspects = {
            'semantic': 'semantic content and meaning',
            'temporal': 'time-related information and sequences', 
            'quantitative': 'numbers, statistics, and measurements',
            'causal': 'cause-effect relationships and reasoning'
        }
        
        encoders = {}
        if self.dense_encoder:
            for aspect_name, aspect_description in aspects.items():
                encoders[aspect_name] = {
                    'description': aspect_description,
                    'prompt_template': f"Extract {aspect_description} from: {{text}}"
                }
        
        return encoders
    
    async def encode_multi_vector(self, text: str, query_type: str = "general") -> Dict[str, np.ndarray]:
        """
        Encode text using multi-vector approach for compositional queries
        Returns vectors for different aspects/components
        """
        vectors = {}
        
        try:
            # Dense semantic encoding
            if self.dense_encoder:
                dense_vector = await self._encode_dense_async(text)
                vectors['dense'] = dense_vector
            
            # Sparse lexical encoding
            sparse_vector = self._encode_sparse(text)
            vectors['sparse'] = sparse_vector
            
            # Aspect-specific encodings for compositional queries
            if query_type in ['analytical', 'comparative', 'temporal']:
                aspect_vectors = await self._encode_aspects(text, query_type)
                vectors.update(aspect_vectors)
                
        except Exception as e:
            logger.error(f"Error in multi-vector encoding: {e}")
            # Fallback to simple dense encoding
            if self.dense_encoder:
                vectors['dense'] = await self._encode_dense_async(text)
        
        return vectors
    
    async def _encode_dense_async(self, text: str) -> np.ndarray:
        """Async dense encoding to avoid blocking"""
        if not self.dense_encoder:
            return np.zeros(self.config.embedding_dimension)
            
        loop = asyncio.get_event_loop()
        
        def encode_sync():
            return self.dense_encoder.encode([text], convert_to_numpy=True)[0]
        
        return await loop.run_in_executor(None, encode_sync)
    
    def _encode_sparse(self, text: str) -> np.ndarray:
        """Encode text using sparse TF-IDF"""
        try:
            sparse_matrix = self.sparse_encoder.transform([text])
            return sparse_matrix.toarray()[0]
        except Exception as e:
            logger.error(f"Sparse encoding error: {e}")
            return np.zeros(self.config.sparse_model_features)
    
    async def _encode_aspects(self, text: str, query_type: str) -> Dict[str, np.ndarray]:
        """Encode text for specific aspects relevant to query type"""
        aspect_vectors = {}
        
        if not self.dense_encoder:
            return aspect_vectors
        
        relevant_aspects = self._get_relevant_aspects(query_type)
        
        for aspect in relevant_aspects:
            if aspect in self.aspect_encoders:
                aspect_text = self._extract_aspect_content(text, aspect)
                aspect_vector = await self._encode_dense_async(aspect_text)
                aspect_vectors[f'aspect_{aspect}'] = aspect_vector
        
        return aspect_vectors
    
    def _get_relevant_aspects(self, query_type: str) -> List[str]:
        """Get aspects relevant to query type"""
        aspect_map = {
            'analytical': ['semantic', 'quantitative'],
            'comparative': ['semantic', 'quantitative'],
            'temporal': ['temporal', 'semantic'],
            'causal': ['causal', 'semantic'],
            'general': ['semantic']
        }
        return aspect_map.get(query_type, ['semantic'])
    
    def _extract_aspect_content(self, text: str, aspect: str) -> str:
        """Extract aspect-specific content from text (simplified)"""
        if aspect == 'quantitative':
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?|\$[\d,]+(?:\.\d+)?[KMB]?', text)
            quantitative_sentences = []
            sentences = text.split('.')
            for sentence in sentences:
                if any(num in sentence for num in numbers) or any(word in sentence.lower() 
                      for word in ['increase', 'decrease', 'growth', 'percent', 'million', 'thousand']):
                    quantitative_sentences.append(sentence.strip())
            return '. '.join(quantitative_sentences) if quantitative_sentences else text
            
        elif aspect == 'temporal':
            temporal_keywords = ['when', 'before', 'after', 'during', 'since', 'until', 'Q1', 'Q2', 'Q3', 'Q4', '2025', '2024']
            temporal_sentences = []
            sentences = text.split('.')
            for sentence in sentences:
                if any(keyword in sentence for keyword in temporal_keywords):
                    temporal_sentences.append(sentence.strip())
            return '. '.join(temporal_sentences) if temporal_sentences else text
            
        elif aspect == 'causal':
            causal_keywords = ['because', 'due to', 'as a result', 'caused by', 'led to', 'resulted in']
            causal_sentences = []
            sentences = text.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in causal_keywords):
                    causal_sentences.append(sentence.strip())
            return '. '.join(causal_sentences) if causal_sentences else text
            
        else:  # semantic (default)
            return text

class EmbeddingQualityAssessor:
    """
    Assess embedding quality for retrieval following Huang et al. (2025)
    Embedding-informed adaptive retrieval methodology
    """
    
    def __init__(self, quality_threshold: float = 0.65):
        self.quality_threshold = quality_threshold
        self.quality_cache = {}
        self.assessment_history = []
        
    async def assess_embedding_quality(self, 
                                     content: str,
                                     query: str,
                                     embedding_vectors: Dict[str, np.ndarray],
                                     retrieval_metadata: Dict[str, Any]) -> float:
        """
        Assess embedding quality using multiple indicators
        Following Huang et al. embedding-informed approach
        """
        cache_key = f"{hash(content)}_{hash(query)}"
        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]
        
        quality_factors = {}
        
        # 1. Vector magnitude consistency
        if 'dense' in embedding_vectors:
            magnitude = np.linalg.norm(embedding_vectors['dense'])
            magnitude_quality = min(magnitude / 10.0, 1.0) if magnitude > 0 else 0.0
            quality_factors['magnitude'] = magnitude_quality
        
        # 2. Multi-vector coherence
        coherence_score = self._assess_multi_vector_coherence(embedding_vectors)
        quality_factors['coherence'] = coherence_score
        
        # 3. Content completeness
        completeness = self._assess_content_completeness(content)
        quality_factors['completeness'] = completeness
        
        # 4. Query relevance
        relevance = self._assess_query_relevance(content, query, retrieval_metadata)
        quality_factors['relevance'] = relevance
        
        # 5. Information density
        density = self._assess_information_density(content)
        quality_factors['density'] = density
        
        # Weighted combination
        weights = {
            'relevance': 0.35,
            'coherence': 0.25,
            'completeness': 0.20,
            'density': 0.15,
            'magnitude': 0.05
        }
        
        overall_quality = sum(weights.get(factor, 0) * score 
                            for factor, score in quality_factors.items())
        
        assessment_record = {
            'timestamp': datetime.utcnow(),
            'content_hash': hash(content),
            'query_hash': hash(query),
            'quality_score': overall_quality,
            'factors': quality_factors,
            'threshold_met': overall_quality >= self.quality_threshold
        }
        self.assessment_history.append(assessment_record)
        
        self.quality_cache[cache_key] = overall_quality
        return overall_quality
    
    def _assess_multi_vector_coherence(self, vectors: Dict[str, np.ndarray]) -> float:
        """Assess coherence between different vector representations"""
        if len(vectors) < 2:
            return 1.0
        
        vector_list = list(vectors.values())
        similarities = []
        
        for i in range(len(vector_list)):
            for j in range(i + 1, len(vector_list)):
                try:
                    v1, v2 = vector_list[i], vector_list[j]
                    if v1.shape != v2.shape:
                        continue
                    similarity = cosine_similarity([v1], [v2])[0][0]
                    similarities.append(similarity)
                except Exception as e:
                    logger.debug(f"Vector coherence calculation error: {e}")
                    continue
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    def _assess_content_completeness(self, content: str) -> float:
        """Assess structural completeness of content"""
        if not content.strip():
            return 0.0
        
        factors = []
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        complete_sentences = len([s for s in sentences if len(s.split()) >= 3])
        sentence_completeness = complete_sentences / len(sentences) if sentences else 0
        factors.append(sentence_completeness)
        
        word_count = len(content.split())
        if word_count < 5:
            length_score = word_count / 5.0
        elif word_count > 200:
            length_score = max(0.5, 1.0 - (word_count - 200) / 400.0)
        else:
            length_score = 1.0
        factors.append(length_score)
        
        punct_count = sum(1 for char in content if char in '.,!?:;')
        word_count = max(len(content.split()), 1)
        punct_ratio = punct_count / word_count
        punct_score = 1.0 if 0.05 <= punct_ratio <= 0.4 else max(0.3, 1.0 - abs(punct_ratio - 0.15) * 2)
        factors.append(punct_score)
        
        return float(np.mean(factors))
    
    def _assess_query_relevance(self, content: str, query: str, metadata: Dict[str, Any]) -> float:
        """Assess content relevance to query"""
        try:
            if 'retrieval_score' in metadata:
                base_relevance = float(metadata['retrieval_score'])
            else:
                query_words = set(query.lower().split())
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                base_relevance = overlap / len(query_words) if query_words else 0.0
            
            content_words = len(content.split())
            if content_words < 10:
                base_relevance *= content_words / 10.0
            
            return min(base_relevance, 1.0)
        except Exception as e:
            logger.debug(f"Query relevance assessment error: {e}")
            return 0.5
    
    def _assess_information_density(self, content: str) -> float:
        """Assess information density of content"""
        words = content.split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        uniqueness = len(unique_words) / len(words)
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'}
        stop_word_count = sum(1 for word in words if word.lower() in stop_words)
        content_word_ratio = 1.0 - (stop_word_count / len(words))
        
        return float(0.6 * uniqueness + 0.4 * content_word_ratio)
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality assessment statistics for monitoring"""
        if not self.assessment_history:
            return {'status': 'no_assessments'}
        
        recent_assessments = [
            a for a in self.assessment_history 
            if a['timestamp'] > datetime.utcnow() - timedelta(hours=24)
        ]
        
        if not recent_assessments:
            return {'status': 'no_recent_assessments'}
        
        quality_scores = [a['quality_score'] for a in recent_assessments]
        threshold_met_count = sum(1 for a in recent_assessments if a['threshold_met'])
        
        return {
            'total_assessments': len(self.assessment_history),
            'recent_assessments_24h': len(recent_assessments),
            'avg_quality_score': float(np.mean(quality_scores)),
            'quality_std': float(np.std(quality_scores)),
            'threshold_pass_rate': threshold_met_count / len(recent_assessments),
            'current_threshold': self.quality_threshold
        }

class HybridVectorStore:
    """
    Hybrid Dense + Sparse Vector Store with Quality Assessment
    Implements multi-vector architecture to address vector bottleneck limitations
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.multi_encoder = MultiVectorEncoder(config)
        self.quality_assessor = EmbeddingQualityAssessor(config.quality_threshold)
        
        # Storage backends
        self.dense_store = self._initialize_dense_store()
        self.sparse_store = self._initialize_sparse_store()
        
        # Document metadata storage
        self.doc_metadata = {}
        self.doc_vectors = {}
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_latency': 0.0,
            'quality_filtered_count': 0
        }
        
        logger.info(f"HybridVectorStore initialized with {config.collection_name}")

    async def initialize(self):
        """Initialize vector store components asynchronously"""
        logger.info("Initializing HybridVectorStore components...")
        try:
            # Warm up dense encoder
            if self.multi_encoder.dense_encoder:
                loop = asyncio.get_event_loop()
                def warmup_dense():
                    self.multi_encoder.dense_encoder.encode(["test sentence"], convert_to_numpy=True)
                await loop.run_in_executor(None, warmup_dense)
                logger.info("Dense encoder warmed up")

            # Ensure persistent storage is ready
            if self.dense_store['type'] == 'chroma' and chromadb:
                client = self.dense_store['client']
                client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"description": "ContextChain v2.0 dense embeddings"}
                )
                logger.info("ChromaDB collection initialized")

            # Load sparse encoder if pre-fitted
            if self.sparse_store['fitted']:
                logger.info("Sparse encoder already fitted")
            else:
                logger.info("Sparse encoder ready for fitting on first indexing")

            logger.info("HybridVectorStore initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HybridVectorStore: {str(e)}")
            raise

    async def close(self):
        """Close resources and persist data"""
        logger.info("Closing HybridVectorStore...")
        try:
            if self.dense_store['type'] == 'chroma' and chromadb:
                client = self.dense_store['client']
                try:
                    client.persist()
                    logger.info("ChromaDB data persisted")
                except Exception as e:
                    logger.warning(f"Failed to persist ChromaDB: {str(e)}")

            # Clear in-memory caches to free resources
            self.doc_metadata.clear()
            self.doc_vectors.clear()
            self.sparse_store['vectors'].clear()
            if self.dense_store['type'] == 'memory':
                self.dense_store['vectors'].clear()

            logger.info("HybridVectorStore closed successfully")
        except Exception as e:
            logger.error(f"Error closing HybridVectorStore: {str(e)}")

    def _initialize_dense_store(self):
        """Initialize dense vector storage (ChromaDB or FAISS)"""
        try:
            if chromadb:
                client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                collection = client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"description": "ContextChain v2.0 dense embeddings"}
                )
                return {'client': client, 'collection': collection, 'type': 'chroma'}
            
            elif faiss:
                index = faiss.IndexFlatIP(self.config.embedding_dimension)
                return {'index': index, 'type': 'faiss', 'doc_ids': []}
            
            else:
                logger.warning("No dense vector backend available, using in-memory storage")
                return {'vectors': {}, 'type': 'memory'}
                
        except Exception as e:
            logger.error(f"Error initializing dense store: {e}")
            return {'vectors': {}, 'type': 'memory'}
    
    def _initialize_sparse_store(self):
        """Initialize sparse vector storage"""
        return {
            'vectorizer': None,
            'vectors': {},
            'fitted': False
        }
    
    async def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        """
        Index documents with multi-vector encoding and quality assessment
        Returns list of document IDs
        """
        logger.info(f"Indexing {len(documents)} documents...")
        doc_ids = []
        
        # Prepare sparse vectorizer training data
        all_texts = [doc.get('content', '') for doc in documents if doc.get('content', '').strip()]
        
        # Fit sparse encoder if not already fitted
        if all_texts and not self.sparse_store['fitted']:
            try:
                self.multi_encoder.sparse_encoder.fit(all_texts)
                self.sparse_store['vectorizer'] = self.multi_encoder.sparse_encoder
                self.sparse_store['fitted'] = True
                logger.info("Sparse encoder fitted successfully")
            except Exception as e:
                logger.error(f"Error fitting sparse encoder: {e}")
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = await self._index_batch(batch)
            doc_ids.extend(batch_ids)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        logger.info(f"Document indexing completed. Total docs: {len(self.doc_metadata)}")
        return doc_ids
    
    async def _index_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Index a batch of documents"""
        batch_ids = []
        
        for doc in documents:
            try:
                doc_id = doc.get('id', str(hash(doc.get('content', ''))))
                content = doc.get('content', '')
                
                if not content.strip():
                    continue
                
                # Multi-vector encoding
                vectors = await self.multi_encoder.encode_multi_vector(
                    content, 
                    query_type='general'
                )
                
                # Store in dense backend
                await self._store_dense_vectors(doc_id, vectors, doc)
                
                # Store in sparse backend
                self._store_sparse_vectors(doc_id, vectors)
                
                # Store metadata
                self.doc_metadata[doc_id] = {
                    'content': content,
                    'metadata': doc.get('metadata', {}),
                    'indexed_at': datetime.utcnow(),
                    'vector_types': list(vectors.keys())
                }
                
                self.doc_vectors[doc_id] = vectors
                batch_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Error indexing document {doc.get('id', 'unknown')}: {e}")
        
        return batch_ids
    
    async def _store_dense_vectors(self, doc_id: str, vectors: Dict[str, np.ndarray], doc: Dict):
        """Store dense vectors in backend"""
        try:
            if self.dense_store['type'] == 'chroma':
                collection = self.dense_store['collection']
                if 'dense' in vectors:
                    embedding = vectors['dense'].tolist()
                    collection.upsert(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[doc.get('content', '')],
                        metadatas=[doc.get('metadata', {})]
                    )
                    
            elif self.dense_store['type'] == 'faiss':
                if 'dense' in vectors:
                    vector = vectors['dense'].astype(np.float32)
                    vector = vector / np.linalg.norm(vector)
                    self.dense_store['index'].add(vector.reshape(1, -1))
                    self.dense_store['doc_ids'].append(doc_id)
                    
            else:
                self.dense_store['vectors'][doc_id] = vectors
                
        except Exception as e:
            logger.error(f"Error storing dense vectors for {doc_id}: {e}")
    
    def _store_sparse_vectors(self, doc_id: str, vectors: Dict[str, np.ndarray]):
        """Store sparse vectors"""
        if 'sparse' in vectors and self.sparse_store['fitted']:
            self.sparse_store['vectors'][doc_id] = vectors['sparse']
    
    async def search(self, 
                    query: str, 
                    k: int = 10,
                    query_type: str = "general",
                    metadata_filter: Optional[Dict[str, Any]] = None,
                    enable_reranking: bool = None) -> List[VectorSearchResult]:
        """
        Hybrid search with quality assessment and fusion
        """
        start_time = datetime.utcnow()
        
        if enable_reranking is None:
            enable_reranking = self.config.enable_reranking
        
        try:
            # Encode query
            query_vectors = await self.multi_encoder.encode_multi_vector(query, query_type)
            
            # Dense search
            dense_results = await self._dense_search(query_vectors, k * 2, metadata_filter)
            
            # Sparse search
            sparse_results = await self._sparse_search(query, k * 2)
            
            # Fusion and reranking
            fused_results = await self._fusion_and_rerank(
                query, query_vectors, dense_results, sparse_results, k, enable_reranking
            )
            
            # Quality filtering
            quality_filtered = await self._apply_quality_filtering(query, fused_results)
            
            # Update statistics
            self._update_retrieval_stats(start_time, len(quality_filtered))
            
            logger.info(f"Search completed: {len(dense_results)} dense + {len(sparse_results)} sparse "
                       f"→ {len(fused_results)} fused → {len(quality_filtered)} final results")
            
            return quality_filtered[:k]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def _dense_search(self, query_vectors: Dict[str, np.ndarray], k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Dense vector search"""
        if 'dense' not in query_vectors:
            return []
        
        query_vector = query_vectors['dense']
        
        try:
            if self.dense_store['type'] == 'chroma':
                collection = self.dense_store['collection']
                results = collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=min(k, collection.count()),
                    where=metadata_filter
                )
                
                search_results = []
                if results['ids'] and results['distances']:
                    for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
                        similarity = max(0.0, 1.0 - distance / 2.0)
                        search_results.append((doc_id, similarity))
                
                return search_results
                
            elif self.dense_store['type'] == 'faiss':
                if self.dense_store['index'].ntotal == 0:
                    return []
                
                norm_query = query_vector.astype(np.float32)
                norm_query = norm_query / np.linalg.norm(norm_query)
                
                k_search = min(k, self.dense_store['index'].ntotal)
                similarities, indices = self.dense_store['index'].search(
                    norm_query.reshape(1, -1), k_search
                )
                
                search_results = []
                for idx, similarity in zip(indices[0], similarities[0]):
                    if idx < len(self.dense_store['doc_ids']):
                        doc_id = self.dense_store['doc_ids'][idx]
                        if metadata_filter:
                            if not self._matches_metadata_filter(doc_id, metadata_filter):
                                continue
                        search_results.append((doc_id, float(similarity)))
                
                return search_results
                
            else:
                if not self.dense_store['vectors']:
                    return []
                
                similarities = []
                for doc_id, doc_vectors in self.dense_store['vectors'].items():
                    if metadata_filter and not self._matches_metadata_filter(doc_id, metadata_filter):
                        continue
                    if 'dense' in doc_vectors:
                        similarity = cosine_similarity([query_vector], [doc_vectors['dense']])[0][0]
                        similarities.append((doc_id, similarity))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:k]
                
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []
    
    def _matches_metadata_filter(self, doc_id: str, metadata_filter: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter"""
        if doc_id not in self.doc_metadata:
            return False
        doc_metadata = self.doc_metadata[doc_id]['metadata']
        for key, value in metadata_filter.items():
            if doc_metadata.get(key) != value:
                return False
        return True
    
    async def _sparse_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Sparse vector search using TF-IDF"""
        if not self.sparse_store['fitted'] or not self.sparse_store['vectors']:
            return []
        
        try:
            query_vector = self.multi_encoder.sparse_encoder.transform([query])
            query_sparse = query_vector.toarray()[0]
            
            similarities = []
            for doc_id, doc_sparse in self.sparse_store['vectors'].items():
                similarity = cosine_similarity([query_sparse], [doc_sparse])[0][0]
                similarities.append((doc_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return []
    
    async def _fusion_and_rerank(self, 
                               query: str,
                               query_vectors: Dict[str, np.ndarray],
                               dense_results: List[Tuple[str, float]],
                               sparse_results: List[Tuple[str, float]],
                               k: int,
                               enable_reranking: bool) -> List[VectorSearchResult]:
        """
        Fusion-in-Decoder style combination of dense and sparse results
        """
        fusion_weights = self.config.fusion_weights or {'dense': 0.7, 'sparse': 0.3}
        all_results = {}
        
        for doc_id, score in dense_results:
            if doc_id in self.doc_metadata:
                all_results[doc_id] = {
                    'dense_score': score,
                    'sparse_score': 0.0,
                    'content': self.doc_metadata[doc_id]['content'],
                    'metadata': self.doc_metadata[doc_id]['metadata']
                }
        
        for doc_id, score in sparse_results:
            if doc_id in self.doc_metadata:
                if doc_id in all_results:
                    all_results[doc_id]['sparse_score'] = score
                else:
                    all_results[doc_id] = {
                        'dense_score': 0.0,
                        'sparse_score': score,
                        'content': self.doc_metadata[doc_id]['content'],
                        'metadata': self.doc_metadata[doc_id]['metadata']
                    }
        
        fused_results = []
        for doc_id, scores in all_results.items():
            fusion_score = (
                fusion_weights['dense'] * scores['dense_score'] +
                fusion_weights['sparse'] * scores['sparse_score']
            )
            
            multi_vector_scores = {}
            if doc_id in self.doc_vectors:
                doc_vectors = self.doc_vectors[doc_id]
                for vector_type, doc_vector in doc_vectors.items():
                    if vector_type in query_vectors:
                        try:
                            similarity = cosine_similarity(
                                [query_vectors[vector_type]], 
                                [doc_vector]
                            )[0][0]
                            multi_vector_scores[vector_type] = similarity
                        except:
                            pass
            
            result = VectorSearchResult(
                content=scores['content'],
                vector_score=scores['dense_score'],
                sparse_score=scores['sparse_score'],
                fusion_score=fusion_score,
                source_id=doc_id,
                metadata=scores['metadata'],
                embedding_quality=0.0,
                multi_vector_scores=multi_vector_scores
            )
            
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.fusion_score, reverse=True)
        
        if enable_reranking and len(fused_results) > 1:
            fused_results = await self._cross_encoder_rerank(query, fused_results[:k*2])
        
        return fused_results[:k]
    
    async def _cross_encoder_rerank(self, query: str, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Cross-encoder reranking for improved relevance"""
        try:
            query_words = set(query.lower().split())
            
            for result in results:
                doc_words = set(result.content.lower().split())
                word_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
                rerank_boost = word_overlap * 0.2
                result.fusion_score = result.fusion_score + rerank_boost
            
            results.sort(key=lambda x: x.fusion_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
        
        return results
    
    async def _apply_quality_filtering(self, query: str, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Apply embedding-informed quality filtering"""
        quality_filtered = []
        
        for result in results:
            try:
                doc_vectors = self.doc_vectors.get(result.source_id, {})
                quality_score = await self.quality_assessor.assess_embedding_quality(
                    result.content,
                    query,
                    doc_vectors,
                    {
                        'retrieval_score': result.fusion_score,
                        'vector_score': result.vector_score,
                        'sparse_score': result.sparse_score
                    }
                )
                
                result.embedding_quality = quality_score
                
                if quality_score >= self.config.quality_threshold:
                    quality_filtered.append(result)
                
            except Exception as e:
                logger.error(f"Quality filtering error for {result.source_id}: {e}")
                result.embedding_quality = 0.5
                quality_filtered.append(result)
        
        self.retrieval_stats['quality_filtered_count'] += len([r for r in results if r not in quality_filtered])
        return quality_filtered
    
    def _update_retrieval_stats(self, start_time: datetime, result_count: int):
        """Update retrieval performance statistics"""
        latency = (datetime.utcnow() - start_time).total_seconds()
        self.retrieval_stats['total_queries'] += 1
        alpha = 0.1
        self.retrieval_stats['avg_latency'] = (
            (1 - alpha) * self.retrieval_stats['avg_latency'] + alpha * latency
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get vector store performance statistics"""
        return {
            'total_documents': len(self.doc_metadata),
            'total_queries': self.retrieval_stats['total_queries'],
            'avg_latency_seconds': self.retrieval_stats['avg_latency'],
            'quality_filter_rate': (
                self.retrieval_stats['quality_filtered_count'] / 
                max(self.retrieval_stats['total_queries'], 1)
            ),
            'dense_store_type': self.dense_store['type'],
            'sparse_store_fitted': self.sparse_store['fitted'],
            'quality_threshold': self.config.quality_threshold,
            'quality_assessor_stats': self.quality_assessor.get_quality_statistics()
        }
    
    async def update_document(self, doc_id: str, updated_doc: Dict[str, Any]):
        """Update existing document"""
        if doc_id not in self.doc_metadata:
            logger.warning(f"Document {doc_id} not found for update")
            return False
        
        try:
            await self._index_batch([{**updated_doc, 'id': doc_id}])
            logger.info(f"Document {doc_id} updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    async def delete_document(self, doc_id: str):
        """Delete document from all stores"""
        if doc_id not in self.doc_metadata:
            logger.warning(f"Document {doc_id} not found for deletion")
            return False
        
        try:
            del self.doc_metadata[doc_id]
            if doc_id in self.doc_vectors:
                del self.doc_vectors[doc_id]
            if doc_id in self.sparse_store['vectors']:
                del self.sparse_store['vectors'][doc_id]
            if self.dense_store['type'] == 'chroma':
                try:
                    self.dense_store['collection'].delete(ids=[doc_id])
                except:
                    pass
            
            logger.info(f"Document {doc_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

async def test_hybrid_vector_store():
    """Test the HybridVectorStore functionality"""
    config = VectorStoreConfig(
        collection_name="test_collection",
        embedding_dimension=384,
        similarity_threshold=0.7,
        quality_threshold=0.65
    )
    
    vector_store = HybridVectorStore(config)
    await vector_store.initialize()
    
    test_docs = [
        {
            'id': 'doc1',
            'content': 'Q3 2025 sales increased by 15% to $2.8M, driven by new product launches and European market expansion.',
            'metadata': {'source': 'sales_report', 'date': '2025-10-01'}
        },
        {
            'id': 'doc2', 
            'content': 'Customer satisfaction improved from 82% to 88% in Q3, with enhanced support and faster delivery times.',
            'metadata': {'source': 'customer_survey', 'date': '2025-09-30'}
        },
        {
            'id': 'doc3',
            'content': 'European expansion generated $400K revenue, with Germany and France as top performing markets.',
            'metadata': {'source': 'market_analysis', 'date': '2025-09-28'}
        }
    ]
    
    await vector_store.index_documents(test_docs)
    
    query = "What drove the sales growth in Q3 2025?"
    results = await vector_store.search(query, k=5, query_type="analytical")
    
    print(f"Search Results for: '{query}'")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Document: {result.source_id}")
        print(f"   Fusion Score: {result.fusion_score:.3f}")
        print(f"   Dense Score: {result.vector_score:.3f}")
        print(f"   Sparse Score: {result.sparse_score:.3f}")
        print(f"   Quality Score: {result.embedding_quality:.3f}")
        print(f"   Content: {result.content[:100]}...")
        print(f"   Multi-Vector Scores: {result.multi_vector_scores}")
    
    stats = vector_store.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Avg Latency: {stats['avg_latency_seconds']:.3f}s")
    print(f"Quality Filter Rate: {stats['quality_filter_rate']:.3f}")
    
    await vector_store.close()
    return results

if __name__ == "__main__":
    asyncio.run(test_hybrid_vector_store())