import asyncio
import pytest
from contextchain.src.dag import DAGEngine, ExecutionContext

@pytest.mark.asyncio
async def test_dag_engine():
    engine = DAGEngine(max_parallel_tasks=5)
    
    context = ExecutionContext(
        session_id='test_session',
        query='Analyze the Q3 2025 sales performance and identify growth drivers',
        raw_data={'source': 'test_data'}
    )
    
    result_context = await engine.execute_workflow('historical_analysis', context)
    
    assert result_context.metadata.get('execution_status') == 'completed'
    assert result_context.metadata.get('total_execution_time') > 0
    assert 'final_response' in result_context.processed_data
    assert 'generated_response' in result_context.processed_data.get('final_response', {})

@pytest.mark.asyncio
async def test_workflow_validation():
    engine = DAGEngine()
    validation = await engine.validate_workflow('historical_analysis')
    
    assert validation['valid']
    assert validation['generated_tasks'] > 0

@pytest.mark.asyncio
async def test_invalid_workflow():
    engine = DAGEngine()
    validation = await engine.validate_workflow('invalid')
    assert not validation['valid']