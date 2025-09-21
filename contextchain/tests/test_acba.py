import asyncio
import pytest
from contextchain.src.acba import AdaptiveContextBudgetingAlgorithm, BudgetAllocation, BudgetArm, QueryComplexity

@pytest.mark.asyncio
async def test_acba_basic_functionality():
    acba = AdaptiveContextBudgetingAlgorithm(max_tokens=2048)
    
    query = "Analyze the sales trends for Q3 2025 and identify key factors driving performance"
    retrieved_docs = [
        {'content': 'Q3 2025 sales increased by 15% compared to Q2...', 'score': 0.95},
        {'content': 'Key drivers include new product launches and market expansion...', 'score': 0.87},
        {'content': 'Regional performance shows strong growth in Asia-Pacific...', 'score': 0.82}
    ]
    
    allocation = await acba.compute_optimal_budget(query, retrieved_docs)
    
    assert isinstance(allocation, BudgetAllocation)
    assert allocation.total_budget <= 2048
    assert allocation.arm_selected in BudgetArm
    assert 0 <= allocation.confidence_score <= 1
    assert allocation.retrieval_tokens + allocation.compression_tokens + allocation.generation_tokens == allocation.total_budget
    
    performance = {
        'accuracy': 0.85,
        'tokens_used': allocation.total_budget - 50,
        'latency_seconds': 1.8
    }
    
    await acba.update_with_feedback(allocation, performance)
    
    summary = acba.get_performance_summary()
    assert 'status' in summary

@pytest.mark.asyncio
async def test_query_complexity_assessment():
    assessor = AdaptiveContextBudgetingAlgorithm().complexity_assessor
    query = "Compare Q3 sales to Q2 and analyze trends"
    complexity = assessor.assess_complexity(query)
    
    assert isinstance(complexity, QueryComplexity)
    assert 0 <= complexity.overall_score <= 1
    assert complexity.semantic_complexity > 0  # Basic assertion

@pytest.mark.asyncio
async def test_rl_compression():
    agent = AdaptiveContextBudgetingAlgorithm().rl_compressor
    docs = ["Document 1 content", "Document 2 content"]
    query = "Test query"
    target_length = 50
    
    compressed, score = await agent.compress_with_rl_optimization(docs, query, target_length)
    
    assert len(compressed) <= target_length + 50  # Approximate check
    assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_edge_cases():
    acba = AdaptiveContextBudgetingAlgorithm()
    
    # Empty query
    with pytest.raises(Exception):
        await acba.compute_optimal_budget("", [])
    
    # No documents
    allocation = await acba.compute_optimal_budget("Simple query", [])
    assert allocation.total_budget > 0
    
    # Feedback with low reward
    performance = {'accuracy': 0.1, 'tokens_used': 1000, 'latency_seconds': 10}
    await acba.update_with_feedback(allocation, performance)