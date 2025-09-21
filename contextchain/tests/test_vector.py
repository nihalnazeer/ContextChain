import asyncio
import pytest
from contextchain.src.vector import HybridVectorStore, VectorStoreConfig

@pytest.mark.asyncio
async def test_hybrid_vector_store():
    config = VectorStoreConfig(
        collection_name="test_collection",
        embedding_dimension=384,
        similarity_threshold=0.7,
        quality_threshold=0.65
    )
    
    vector_store = HybridVectorStore(config)
    
    test_docs = [
        {'id': 'doc1', 'content': 'Test content 1', 'metadata': {}},
        {'id': 'doc2', 'content': 'Test content 2', 'metadata': {}}
    ]
    
    await vector_store.index_documents(test_docs)
    
    query = "Test query"
    results = await vector_store.search(query, k=2, query_type="general")
    
    assert len(results) <= 2
    assert all(0 <= r.embedding_quality <= 1 for r in results)
    
    stats = vector_store.get_performance_stats()
    assert stats['total_documents'] == 2