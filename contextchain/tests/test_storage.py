import asyncio
import pytest
from contextchain.src.storage import IntelligentStorage

@pytest.mark.asyncio
async def test_storage_mongodb():
    storage = IntelligentStorage(mongo_uri="mongodb://localhost:27017/contextchain_test")
    await storage.initialize()
    
    # Test logging
    await storage.log_interaction(
        session_id="test_session",
        query="Test query",
        complexity={"overall_score": 0.5},
        budget_allocation={"total_budget": 2000},
        performance_metrics={"latency": 1.2, "tokens_used": 150},
        success=True
    )
    
    # Test feedback update
    updated = await storage.update_user_feedback("test_session", 5, "Great response!")
    assert updated
    
    # Test analytics
    analytics = await storage.get_performance_dashboard_data(days_back=1)
    assert analytics['total_interactions'] >= 1
    
    storage.close()

@pytest.mark.asyncio
async def test_storage_sqlite():
    storage = IntelligentStorage(sqlite_path=":memory:")  # In-memory for testing
    await storage.initialize()
    
    # Similar tests as above
    await storage.log_interaction(...)  # Test log
    updated = await storage.update_user_feedback(...)
    assert updated
    analytics = await storage.get_performance_dashboard_data(...)
    assert 'total_interactions' in analytics