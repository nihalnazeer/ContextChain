import asyncio
import pytest
from contextchain.src.context_engineer import ContextEngineer, PromptConstructionConfig, PromptStyle, BudgetAllocation, BudgetArm

@pytest.mark.asyncio
async def test_context_engineer():
    config = PromptConstructionConfig(
        style=PromptStyle.ANALYTICAL,
        max_prompt_tokens=1500,
        include_metadata=True,
        include_provenance=True,
        safety_filtering=True
    )
    
    engineer = ContextEngineer(config)
    
    query = "Analyze the Q3 2025 sales performance and identify key growth drivers"
    
    raw_docs = [
        {
            'content': 'Q3 2025 sales reached $2.8M, representing a 22% increase from Q2 2025 ($2.3M). The primary growth drivers were the launch of our premium product line, expansion into the European market, and improved customer retention rates.',
            'source_id': 'sales_report_q3_2025',
            'score': 0.95,
            'authority_score': 0.9,
            'temporal_relevance': 0.95
        },
        {
            'content': 'European market expansion contributed $400K to Q3 revenue. Customer acquisition cost decreased by 15% due to improved marketing efficiency. Premium product line generated 30% of total revenue.',
            'source_id': 'market_analysis_europe',
            'score': 0.88,
            'authority_score': 0.85,
            'temporal_relevance': 0.9
        },
        {
            'content': 'Customer retention improved from 78% in Q2 to 85% in Q3. This improvement was attributed to enhanced customer support and product quality improvements.',
            'source_id': 'customer_metrics_q3',
            'score': 0.82,
            'authority_score': 0.8,
            'temporal_relevance': 0.92
        }
    ]
    
    budget = BudgetAllocation(
        retrieval_tokens=800,
        compression_tokens=200,
        generation_tokens=500,
        total_budget=1500,
        arm_selected=BudgetArm.ADAPTIVE_COMPRESS,
        confidence_score=0.85,
        hierarchy_weights={},
        expected_utility=0.78,
        allocation_timestamp=datetime.utcnow()
    )
    
    prompt = engineer.build_prompt(query, raw_docs, budget)
    
    assert len(prompt) > 0
    assert "QUERY" in prompt  # Check for structure
    assert "CONTEXT" in prompt
    assert "INSTRUCTIONS" in prompt or "ANALYTICAL FRAMEWORK" in prompt  # Style-specific

@pytest.mark.asyncio
async def test_safety_filtering():
    config = PromptConstructionConfig(safety_filtering=True)
    engineer = ContextEngineer(config)
    
    # Test PII redaction
    raw_docs = [{'content': 'User email: test@example.com, phone: 123-456-7890'}]
    prompt = engineer.build_prompt("Test", raw_docs, BudgetAllocation(...) )
    assert "EMAIL_REDACTED" in prompt
    assert "PHONE_REDACTED" in prompt

@pytest.mark.asyncio
async def test_fallback_prompt():
    engineer = ContextEngineer()
    query = "Fallback test"
    raw_docs = [{'content': 'Test doc'}]
    prompt = engineer._build_fallback_prompt(query, raw_docs)
    assert "Context" in prompt
    assert "Query" in prompt