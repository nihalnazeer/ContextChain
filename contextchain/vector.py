"""
Advanced Vector Store with Multi-Vector Architecture
Addressing: Vector Bottleneck (2025), Embedding-Informed Adaptive Retrieval (Huang et al.)
- Multi-vector dense and sparse retrieval
- Embedding quality assessment and fusion
- Persistent storage support via FAISS (preferred embedded), Chroma, or memory
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
import logger

# Optional imports with fallbacks
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    logger.warning("ChromaDB not available; falling back to other backends")

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not available; falling back to memory backend")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("SentenceTransformers not available; dense encoding disabled")

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
    """Configuration for vector store"""
    backend: str = "faiss"  # 'faiss' (embedded/file), 'chroma' (persistent), 'memory'
    dense_model_name: str = "all-MiniLM-L6-v2"
    sparse_model_features: int = 10000
    collection_name: str = "contextchain_docs"
    persist_directory: str = "./vector_db"
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    fusion_weights: Dict[str, float] = None  # Default: {'dense': 0.7, 'sparse': 0.3}
    enable_reranking: bool = True
    enable_sparse: bool = False  # Toggle sparse for lighter mode
    quality_threshold: float = 0.65
    max_results: int = 100

class MultiVectorEncoder:
    """Multi-vector encoding to address compositional query limitations"""
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dense_encoder = self._initialize_dense_encoder()
        self.sparse_encoder = None
        self.aspect_encoders = self._initialize_aspect_encoders()

    def _initialize_dense_encoder(self):
        if SentenceTransformer is None:
            return None
        try:
            model = SentenceTransformer(self.config.dense_model_name)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error initializing dense encoder: {e}")
            return None

    def _initialize_sparse_encoder(self):
        return TfidfVectorizer(
            max_features=self.config.sparse_model_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

    def _initialize_aspect_encoders(self):
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
        vectors = {}
        try:
            if self.dense_encoder:
                dense_vector = await self._encode_dense_async(text)
                vectors['dense'] = dense_vector
            #if self.sparse_encoder:
                #sparse_vector = self._encode_sparse(text)
                #vectors['sparse'] = sparse_vector
            if query_type in ['analytical', 'comparative', 'temporal']:
                aspect_vectors = await self._encode_aspects(text, query_type)
                vectors.update(aspect_vectors)
        except Exception as e:
            logger.error(f"Error in multi-vector encoding: {e}")
            if self.dense_encoder:
                vectors['dense'] = await self._encode_dense_async(text)
        return vectors

    async def _encode_dense_async(self, text: str) -> np.ndarray:
        if not self.dense_encoder:
            return np.zeros(self.config.embedding_dimension)
        loop = asyncio.get_event_loop()
        def encode_sync():
            return self.dense_encoder.encode([text], convert_to_numpy=True)[0]
        return await loop.run_in_executor(None, encode_sync)

    def _encode_sparse(self, text: str) -> np.ndarray:
        try:
            return self.sparse_encoder.transform([text]).toarray()[0]
        except Exception as e:
            logger.error(f"Sparse encoding error: {e}")
            return np.zeros(self.config.sparse_model_features)

    async def _encode_aspects(self, text: str, query_type: str) -> Dict[str, np.ndarray]:
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
        aspect_map = {
            'analytical': ['semantic', 'quantitative'],
            'comparative': ['semantic', 'quantitative'],
            'temporal': ['temporal', 'semantic'],
            'causal': ['causal', 'semantic'],
            'general': ['semantic']
        }
        return aspect_map.get(query_type, ['semantic'])

    def _extract_aspect_content(self, text: str, aspect: str) -> str:
        import re
        if aspect == 'quantitative':
            numbers = re.findall(r'\d+(?:\.\d+)?%?|\$[\d,]+(?:\.\d+)?[KMB]?', text)
            sentences = text.split('.')
            quantitative_sentences = [s.strip() for s in sentences if any(num in s for num in numbers) or any(word in s.lower() for word in ['increase', 'decrease', 'growth', 'percent', 'million', 'thousand'])]
            return '. '.join(quantitative_sentences) if quantitative_sentences else text
        elif aspect == 'temporal':
            temporal_keywords = ['when', 'before', 'after', 'during', 'since', 'until', 'Q1', 'Q2', 'Q3', 'Q4', '2025', '2024']
            sentences = text.split('.')
            temporal_sentences = [s.strip() for s in sentences if any(keyword in s for keyword in temporal_keywords)]
            return '. '.join(temporal_sentences) if temporal_sentences else text
        elif aspect == 'causal':
            causal_keywords = ['because', 'due to', 'as a result', 'caused by', 'led to', 'resulted in']
            sentences = text.split('.')
            causal_sentences = [s.strip() for s in sentences if any(keyword in s.lower() for keyword in causal_keywords)]
            return '. '.join(causal_sentences) if causal_sentences else text
        else:
            return text

class EmbeddingQualityAssessor:
    """Assess embedding quality for retrieval following Huang et al. (2025)"""
    def __init__(self, quality_threshold: float = 0.65):
        self.quality_threshold = quality_threshold
        self.quality_cache = {}
        self.assessment_history = []

    async def assess_embedding_quality(self, content: str, query: str, embedding_vectors: Dict[str, np.ndarray], retrieval_metadata: Dict[str, Any]) -> float:
        cache_key = f"{hash(content)}_{hash(query)}"
        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]
        quality_factors = {}
        if 'dense' in embedding_vectors:
            magnitude = np.linalg.norm(embedding_vectors['dense'])
            magnitude_quality = min(magnitude / 10.0, 1.0) if magnitude > 0 else 0.0
            quality_factors['magnitude'] = magnitude_quality
        coherence_score = self._assess_multi_vector_coherence(embedding_vectors)
        quality_factors['coherence'] = coherence_score
        completeness = self._assess_content_completeness(content)
        quality_factors['completeness'] = completeness
        relevance = self._assess_query_relevance(content, query, retrieval_metadata)
        quality_factors['relevance'] = relevance
        density = self._assess_information_density(content)
        quality_factors['density'] = density
        weights = {'relevance': 0.35, 'coherence': 0.25, 'completeness': 0.20, 'density': 0.15, 'magnitude': 0.05}
        overall_quality = sum(weights.get(factor, 0) * score for factor, score in quality_factors.items())
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
        if len(vectors) < 2:
            return 1.0
        vector_list = list(vectors.values())
        similarities = []
        for i in range(len(vector_list)):
            for j in range(i + 1, len(vector_list)):
                v1, v2 = vector_list[i], vector_list[j]
                if v1.shape != v2.shape:
                    continue
                similarity = cosine_similarity([v1], [v2])[0][0]
                similarities.append(similarity)
        return float(np.mean(similarities)) if similarities else 0.5

    def _assess_content_completeness(self, content: str) -> float:
        if not content.strip():
            return 0.0
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        complete_sentences = len([s for s in sentences if len(s.split()) >= 3])
        sentence_completeness = complete_sentences / len(sentences) if sentences else 0
        word_count = len(content.split())
        length_score = min(word_count / 5.0, 1.0) if word_count < 5 else min(1.0, max(0.5, 1.0 - (word_count - 200) / 400.0)) if word_count > 200 else 1.0
        punct_count = sum(1 for char in content if char in '.,!?:;')
        punct_ratio = punct_count / max(word_count, 1)
        punct_score = 1.0 if 0.05 <= punct_ratio <= 0.4 else max(0.3, 1.0 - abs(punct_ratio - 0.15) * 2)
        return float(np.mean([sentence_completeness, length_score, punct_score]))

    def _assess_query_relevance(self, content: str, query: str, metadata: Dict[str, Any]) -> float:
        if 'retrieval_score' in metadata:
            base_relevance = float(metadata['retrieval_score'])
        else:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            base_relevance = overlap / len(query_words) if query_words else 0.0
        word_count = len(content.split())
        if word_count < 10:
            base_relevance *= word_count / 10.0
        return min(base_relevance, 1.0)

    def _assess_information_density(self, content: str) -> float:
        words = content.split()
        if not words:
            return 0.0
        unique_words = set(words)
        uniqueness = len(unique_words) / len(words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'}
        stop_word_count = sum(1 for word in words if word.lower() in stop_words)
        content_word_ratio = 1.0 - (stop_word_count / len(words))
        return float(0.6 * uniqueness + 0.4 * content_word_ratio)

    def get_quality_statistics(self) -> Dict[str, Any]:
        if not self.assessment_history:
            return {'status': 'no_assessments'}
        recent_assessments = [a for a in self.assessment_history if a['timestamp'] > datetime.utcnow() - timedelta(hours=24)]
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
    """Hybrid Dense + Sparse Vector Store with Quality Assessment"""
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.multi_encoder = MultiVectorEncoder(config)
        self.quality_assessor = EmbeddingQualityAssessor(config.quality_threshold)
        self.dense_store = self._initialize_dense_store()
        self.sparse_store = None
        self.doc_metadata = {}
        self.doc_vectors = {}
        self.retrieval_stats = {'total_queries': 0, 'avg_latency': 0.0, 'quality_filtered_count': 0}
        logger.info(f"HybridVectorStore initialized with backend: {config.backend}, collection: {config.collection_name}")

    def _initialize_dense_store(self):
        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        if self.config.backend == 'chroma' and chromadb:
            client = chromadb.PersistentClient(path=str(persist_path), settings=Settings(anonymized_telemetry=False))
            collection = client.get_or_create_collection(name=self.config.collection_name)
            return {'client': client, 'collection': collection, 'type': 'chroma'}
        elif self.config.backend == 'faiss' and faiss:
            index_path = persist_path / f"{self.config.collection_name}.faiss"
            metadata_path = persist_path / f"{self.config.collection_name}_metadata.pkl"
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    doc_ids = pickle.load(f)
            else:
                index = faiss.IndexFlatIP(self.config.embedding_dimension)
                doc_ids = []
            return {'index': index, 'type': 'faiss', 'doc_ids': doc_ids, 'index_path': index_path, 'metadata_path': metadata_path}
        else:
            logger.warning(f"Using in-memory storage (backend: {self.config.backend})")
            metadata_path = persist_path / f"{self.config.collection_name}_memory.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    vectors = pickle.load(f)
            else:
                vectors = {}
            return {'vectors': vectors, 'type': 'memory', 'metadata_path': metadata_path}

    def _initialize_sparse_store(self):
        persist_path = Path(self.config.persist_directory) / f"{self.config.collection_name}_sparse.pkl"
        if self.config.enable_sparse:
            if persist_path.exists():
                with open(persist_path, 'rb') as f:
                    data = pickle.load(f)
                vectorizer = data['vectorizer']
                vectors = data['vectors']
                fitted = True
            else:
                vectorizer = self.multi_encoder.sparse_encoder
                vectors = {}
                fitted = False
            return {'vectorizer': vectorizer, 'vectors': vectors, 'fitted': fitted, 'persist_path': persist_path}
        return None

    async def initialize(self):
        logger.info("Initializing HybridVectorStore components...")
        try:
            if self.multi_encoder.dense_encoder:
                loop = asyncio.get_event_loop()
                def warmup_dense():
                    self.multi_encoder.dense_encoder.encode(["test sentence"], convert_to_numpy=True)
                await loop.run_in_executor(None, warmup_dense)
            if self.sparse_store:
                logger.info("Sparse encoder ready")
            logger.info("HybridVectorStore initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

    async def close(self):
        logger.info("Closing HybridVectorStore...")
        try:
            if self.dense_store['type'] == 'faiss':
                faiss.write_index(self.dense_store['index'], str(self.dense_store['index_path']))
                with open(self.dense_store['metadata_path'], 'wb') as f:
                    pickle.dump(self.dense_store['doc_ids'], f)
            elif self.dense_store['type'] == 'memory':
                with open(self.dense_store['metadata_path'], 'wb') as f:
                    pickle.dump(self.dense_store['vectors'], f)
            if self.sparse_store:
                data = {'vectorizer': self.sparse_store['vectorizer'], 'vectors': self.sparse_store['vectors']}
                with open(self.sparse_store['persist_path'], 'wb') as f:
                    pickle.dump(data, f)
            self.doc_metadata.clear()
            self.doc_vectors.clear()
            logger.info("Closed successfully")
        except Exception as e:
            logger.error(f"Error closing: {str(e)}")

    async def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        logger.info(f"Indexing {len(documents)} documents...")
        doc_ids = []
        # REMOVE sparse fitting logic since sparse_store is None
        # all_texts = [doc.get('content', '') for doc in documents if doc.get('content', '').strip()]
        # if all_texts and self.sparse_store and not self.sparse_store['fitted']:
        #     try:
        #         self.sparse_store['vectorizer'].fit(all_texts)
        #         self.sparse_store['fitted'] = True
        #         logger.info("Sparse encoder fitted")
        #     except Exception as e:
        #         logger.error(f"Error fitting sparse: {e}")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = await self._index_batch(batch)
            doc_ids.extend(batch_ids)
        return doc_ids

    async def _index_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        batch_ids = []
        for doc in documents:
            try:
                doc_id = doc.get('id', str(hash(doc.get('content', ''))))
                content = doc.get('content', '')
                if not content.strip():
                    continue
                vectors = await self.multi_encoder.encode_multi_vector(content, 'general')
                await self._store_dense_vectors(doc_id, vectors, doc)
                if self.sparse_store:
                    self._store_sparse_vectors(doc_id, vectors)
                self.doc_metadata[doc_id] = {
                    'content': content,
                    'metadata': doc.get('metadata', {}),
                    'indexed_at': datetime.utcnow(),
                    'vector_types': list(vectors.keys())
                }
                self.doc_vectors[doc_id] = vectors
                batch_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error indexing doc {doc.get('id', 'unknown')}: {e}")
        return batch_ids

    async def _store_dense_vectors(self, doc_id: str, vectors: Dict[str, np.ndarray], doc: Dict):
        if 'dense' not in vectors:
            return
        vector = vectors['dense']
        if self.dense_store['type'] == 'chroma':
            self.dense_store['collection'].upsert(ids=[doc_id], embeddings=[vector.tolist()], documents=[doc['content']], metadatas=[doc.get('metadata', {})])
        elif self.dense_store['type'] == 'faiss':
            norm_vector = vector.astype(np.float32) / np.linalg.norm(vector)
            self.dense_store['index'].add(norm_vector.reshape(1, -1))
            self.dense_store['doc_ids'].append(doc_id)
        else:
            self.dense_store['vectors'][doc_id] = vectors

    def _store_sparse_vectors(self, doc_id: str, vectors: Dict[str, np.ndarray]):
        if 'sparse' in vectors:
            self.sparse_store['vectors'][doc_id] = vectors['sparse']

    async def search(self, query: str, k: int = 10, query_type: str = "general", metadata_filter: Optional[Dict[str, Any]] = None, enable_reranking: bool = None) -> List[VectorSearchResult]:
        start_time = datetime.utcnow()
        if enable_reranking is None:
            enable_reranking = self.config.enable_reranking
        try:
            query_vectors = await self.multi_encoder.encode_multi_vector(query, query_type)
            dense_results = await self._dense_search(query_vectors, k * 2, metadata_filter)
            sparse_results = await self._sparse_search(query, k * 2, metadata_filter) if self.sparse_store else []
            fused_results = await self._fusion_and_rerank(query, query_vectors, dense_results, sparse_results, k, enable_reranking)
            quality_filtered = await self._apply_quality_filtering(query, fused_results)
            self._update_retrieval_stats(start_time, len(quality_filtered))
            return quality_filtered[:min(k, self.config.max_results)]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def _dense_search(self, query_vectors: Dict[str, np.ndarray], k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        if 'dense' not in query_vectors:
            return []
        query_vector = query_vectors['dense']
        if self.dense_store['type'] == 'chroma':
            results = self.dense_store['collection'].query(query_embeddings=[query_vector.tolist()], n_results=k, where=metadata_filter)
            search_results = []
            if results['ids'] and results['distances']:
                for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
                    similarity = max(0.0, 1.0 - distance / 2.0)
                    search_results.append((doc_id, similarity))
            return search_results
        elif self.dense_store['type'] == 'faiss':
            if self.dense_store['index'].ntotal == 0:
                return []
            norm_query = query_vector.astype(np.float32) / np.linalg.norm(query_vector)
            similarities, indices = self.dense_store['index'].search(norm_query.reshape(1, -1), min(k, self.dense_store['index'].ntotal))
            search_results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1 or idx >= len(self.dense_store['doc_ids']):
                    continue
                doc_id = self.dense_store['doc_ids'][idx]
                if metadata_filter and not self._matches_metadata_filter(doc_id, metadata_filter):
                    continue
                search_results.append((doc_id, float(similarity)))
            return search_results
        else:
            if not self.dense_store['vectors']:
                return []
            similarities = [(doc_id, cosine_similarity([query_vector], [doc_vectors['dense']])[0][0]) for doc_id, doc_vectors in self.dense_store['vectors'].items() if 'dense' in doc_vectors and (not metadata_filter or self._matches_metadata_filter(doc_id, metadata_filter))]
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]

    async def _sparse_search(self, query: str, k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        if not self.sparse_store or not self.sparse_store['fitted'] or not self.sparse_store['vectors']:
            return []
        query_vector = self.sparse_store['vectorizer'].transform([query]).toarray()[0]
        similarities = [(doc_id, cosine_similarity([query_vector], [doc_sparse])[0][0]) for doc_id, doc_sparse in self.sparse_store['vectors'].items() if not metadata_filter or self._matches_metadata_filter(doc_id, metadata_filter)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _matches_metadata_filter(self, doc_id: str, metadata_filter: Dict[str, Any]) -> bool:
        if doc_id not in self.doc_metadata:
            return False
        doc_meta = self.doc_metadata[doc_id]['metadata']
        return all(doc_meta.get(key) == value for key, value in metadata_filter.items())

    async def _fusion_and_rerank(self, query: str, query_vectors: Dict[str, np.ndarray], dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]], k: int, enable_reranking: bool) -> List[VectorSearchResult]:
        # UPDATE: Sparse is disabled, use dense-only weights
        fusion_weights = self.config.fusion_weights or {'dense': 1.0, 'sparse': 0.0}
        
        all_results = {}
        for doc_id, score in dense_results:
            if doc_id in self.doc_metadata:
                all_results[doc_id] = {'dense_score': score, 'sparse_score': 0.0, 'content': self.doc_metadata[doc_id]['content'], 'metadata': self.doc_metadata[doc_id]['metadata']}
        
        # SKIP sparse_results processing since sparse_store is None
        # for doc_id, score in sparse_results:
        #     if doc_id in self.doc_metadata:
        #         if doc_id in all_results:
        #             all_results[doc_id]['sparse_score'] = score
        #         else:
        #             all_results[doc_id] = {'dense_score': 0.0, 'sparse_score': score, 'content': self.doc_metadata[doc_id]['content'], 'metadata': self.doc_metadata[doc_id]['metadata']}
        
        fused_results = []
        for doc_id, scores in all_results.items():
            # Now sparse_score is always 0.0, dense_score gets full weight
            fusion_score = fusion_weights['dense'] * scores['dense_score'] + fusion_weights['sparse'] * scores['sparse_score']
            multi_vector_scores = {}
            if doc_id in self.doc_vectors:
                doc_vectors = self.doc_vectors[doc_id]
                for vector_type, doc_vector in doc_vectors.items():
                    if vector_type in query_vectors:
                        similarity = cosine_similarity([query_vectors[vector_type]], [doc_vector])[0][0]
                        multi_vector_scores[vector_type] = similarity
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
        try:
            query_words = set(query.lower().split())
            for result in results:
                doc_words = set(result.content.lower().split())
                word_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
                rerank_boost = word_overlap * 0.2
                result.fusion_score += rerank_boost
            results.sort(key=lambda x: x.fusion_score, reverse=True)
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
        return results

    async def _apply_quality_filtering(self, query: str, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        quality_filtered = []
        for result in results:
            doc_vectors = self.doc_vectors.get(result.source_id, {})
            quality_score = await self.quality_assessor.assess_embedding_quality(
                result.content, query, doc_vectors, {'retrieval_score': result.fusion_score, 'vector_score': result.vector_score, 'sparse_score': result.sparse_score}
            )
            result.embedding_quality = quality_score
            if quality_score >= self.config.quality_threshold:
                quality_filtered.append(result)
        self.retrieval_stats['quality_filtered_count'] += len(results) - len(quality_filtered)
        return quality_filtered

    def _update_retrieval_stats(self, start_time: datetime, result_count: int):
        latency = (datetime.utcnow() - start_time).total_seconds()
        self.retrieval_stats['total_queries'] += 1
        alpha = 0.1
        self.retrieval_stats['avg_latency'] = (1 - alpha) * self.retrieval_stats['avg_latency'] + alpha * latency

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': len(self.doc_metadata),
            'total_queries': self.retrieval_stats['total_queries'],
            'avg_latency_seconds': self.retrieval_stats['avg_latency'],
            'quality_filter_rate': self.retrieval_stats['quality_filtered_count'] / max(self.retrieval_stats['total_queries'], 1),
            'dense_store_type': self.dense_store['type'],
            'sparse_enabled': False,  # Changed from bool(self.sparse_store)
            'quality_threshold': self.config.quality_threshold,
            'quality_stats': self.quality_assessor.get_quality_statistics()
        }

    async def update_document(self, doc_id: str, updated_doc: Dict[str, Any]):
        if doc_id not in self.doc_metadata:
            logger.warning(f"Document {doc_id} not found")
            return False
        await self.delete_document(doc_id)
        await self._index_batch([updated_doc])
        return True

    async def delete_document(self, doc_id: str):
        if doc_id not in self.doc_metadata:
            return False
        del self.doc_metadata[doc_id]
        if doc_id in self.doc_vectors:
            del self.doc_vectors[doc_id]
        if self.sparse_store and doc_id in self.sparse_store['vectors']:
            del self.sparse_store['vectors'][doc_id]
        if self.dense_store['type'] == 'chroma':
            self.dense_store['collection'].delete(ids=[doc_id])
        elif self.dense_store['type'] == 'faiss':
            # FAISS doesn't support delete; rebuild index (inefficient for large, but simple)
            new_index = faiss.IndexFlatIP(self.config.embedding_dimension)
            new_doc_ids = [did for did in self.dense_store['doc_ids'] if did != doc_id]
            for did in new_doc_ids:
                vector = self.doc_vectors[did]['dense']
                norm_vector = vector.astype(np.float32) / np.linalg.norm(vector)
                new_index.add(norm_vector.reshape(1, -1))
            self.dense_store['index'] = new_index
            self.dense_store['doc_ids'] = new_doc_ids
        else:
            if doc_id in self.dense_store['vectors']:
                del self.dense_store['vectors'][doc_id]
        return True

async def test_hybrid_vector_store():
    config = VectorStoreConfig(backend='memory')  # Safe for testing
    vector_store = HybridVectorStore(config)
    await vector_store.initialize()
    test_docs = [
        {'id': 'doc1', 'content': 'Q3 2025 sales increased by 15% to $2.8M, driven by new product launches and European market expansion.', 'metadata': {'source': 'sales_report', 'date': '2025-10-01'}},
        {'id': 'doc2', 'content': 'Customer satisfaction improved from 82% to 88% in Q3, with enhanced support and faster delivery times.', 'metadata': {'source': 'customer_survey', 'date': '2025-09-30'}},
        {'id': 'doc3', 'content': 'European expansion generated $400K revenue, with Germany and France as top performing markets.', 'metadata': {'source': 'market_analysis', 'date': '2025-09-28'}}
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