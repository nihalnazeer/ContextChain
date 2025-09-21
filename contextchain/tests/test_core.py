import asyncio
import pytest
from contextchain.src.core import ContextChain, ContextChainConfig, ContextChainResponse

@pytest.mark.asyncio
async def test_core_orchestrator():
    config = ContextChainConfig(
        max_tokens=2048,
        mongo_uri="mongodb://localhost:27017/contextchain_test"
    )
    
    cc = ContextChain(config)
    await cc.initialize()
    
    query = "Test query for core"
    response = await cc.process_query(query)
    
    assert isinstance(response, ContextChainResponse)
    assert response.success
    assert response.response is not None
    assert response.total_latency_seconds > 0
    assert len(response.processing_steps) > 0
    
    # Test error handling
    with pytest.raises(ValueError):
        await cc.process_query("")  # Empty query

@pytest.mark.asyncio
async def test_index_documents():
    cc = ContextChain(ContextChainConfig())
    await cc.initialize()
    
    docs = [{'content': 'Test doc', 'metadata': {}}]
    index_result = await cc.index_documents(docs)
    
    assert index_result['status'] == 'success'
    assert index_result['documents_indexed'] == 1

@pytest.mark.asyncio
async def test_system_status():
    cc = ContextChain(ContextChainConfig())
    await cc.initialize()
    
    status = await cc.get_system_status()
    
    assert status['status'] == 'healthy'
    assert 'components' in status