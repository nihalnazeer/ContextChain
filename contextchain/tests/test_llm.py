import asyncio
import pytest
from contextchain.src.llm import LLMOptimizer, LLMConfig, LLMBackend, BudgetAllocation, BudgetArm
import datetime as datetime

@pytest.mark.asyncio
async def test_llm_optimizer_openai():
    config = LLMConfig(
        backend=LLMBackend.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="test_key"  # Use mock or env var in real tests
    )
    optimizer = LLMOptimizer(config)
    
    health = await optimizer.health_check()
    assert health['status'] == 'healthy'  # Adjust based on actual check
    
    prompt = "Test prompt"
    budget = BudgetAllocation(retrieval_tokens=100, compression_tokens=100, generation_tokens=300, total_budget=500,
                              arm_selected=BudgetArm.LIGHT_RETRIEVE, confidence_score=0.9, hierarchy_weights={}, expected_utility=0.8, allocation_timestamp=datetime.now())
    
    result = await optimizer.generate_optimized(prompt, budget)
    assert result.content is not None
    assert result.tokens_used > 0

# Similar tests for other backends (Anthropic, Ollama, HuggingFace)
# Note: For real tests, use mocking libraries like pytest-mock or httpx for API calls

@pytest.mark.asyncio
async def test_llm_huggingface():
    config = LLMConfig(
        backend=LLMBackend.HUGGINGFACE,
        model_name="gpt2",
        device="cpu"
    )
    optimizer = LLMOptimizer(config)
    
    prompt = "Test prompt"
    budget = BudgetAllocation(...)  # As above
    
    result = await optimizer.generate_optimized(prompt, budget)
    assert "Error" not in result.content  # Check for successful generation