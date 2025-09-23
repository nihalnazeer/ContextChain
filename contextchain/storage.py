"""
IntelligentStorage for ContextChain v2.0
- Manages MongoDB client lifecycle with auto-start of local MongoDB if needed
- Creates and maintains necessary collections for interactions, metadata, and feedback
- Provides async methods for storing and querying data
- Designed for seamless integration with ContextChain core.py and API
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

import motor.motor_asyncio
from pymongo.errors import ServerSelectionTimeoutError, CollectionInvalid

logger = logging.getLogger(__name__)

def to_mongodb_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to MongoDB-serializable format.
    Handles Enum, datetime, and nested dataclasses/Pydantic models.
    """
    if isinstance(obj, Enum):
        return obj.value if hasattr(obj, 'value') else str(obj.name)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
        # Convert dataclasses, Pydantic models, or objects with __dict__
        return {k: to_mongodb_serializable(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [to_mongodb_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_mongodb_serializable(v) for k, v in obj.items()}
    else:
        return obj

class IntelligentStorage:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/contextchain"):
        self.mongo_uri = mongo_uri
        self.mongo_client = None
        self.db = None
        self.mongod_process: Optional[subprocess.Popen] = None
        self._started_local_mongo = False

    async def initialize(self):
        """Initialize MongoDB connection, auto-start local mongod if needed"""
        try:
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            await self.mongo_client.server_info()  # Test connection
            self.db = self.mongo_client.get_default_database()
            logger.info(f"Connected to MongoDB at {self.mongo_uri}")
            await self._ensure_collections()
        except ServerSelectionTimeoutError:
            logger.warning("MongoDB not reachable, attempting to start a local mongod instance")
            await self._start_local_mongod()
            await asyncio.sleep(5)  # Wait for mongod to stabilize
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            await self.mongo_client.server_info()
            self.db = self.mongo_client.get_default_database()
            logger.info(f"Local MongoDB started and connected at {self.mongo_uri}")
            await self._ensure_collections()
            self._started_local_mongo = True
        except Exception as e:
            logger.error(f"MongoDB initialization failed: {str(e)}")
            raise

    async def _start_local_mongod(self):
        """Start local mongod process with a temporary data directory"""
        data_dir = Path("/tmp/contextchain_mongodb")
        data_dir.mkdir(parents=True, exist_ok=True)
        self.mongod_process = subprocess.Popen([
            "mongod",
            "--dbpath", str(data_dir),
            "--port", "27017",
            "--bind_ip", "127.0.0.1",
            "--quiet"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Started local mongod process")

    async def _ensure_collections(self):
        """Ensure necessary collections exist with indexes"""
        # Get existing collections to avoid duplicate creation
        collections = await self.db.list_collection_names()

        # Create collections only if they don't exist
        for collection_name in ["interactions", "metadata", "feedback"]:
            if collection_name not in collections:
                try:
                    await self.db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
                except CollectionInvalid as e:
                    logger.warning(f"Collection {collection_name} creation skipped (already exists): {str(e)}")

        # Create indexes for interactions
        await self.db.interactions.create_index("session_id")
        await self.db.interactions.create_index("timestamp")
        logger.debug("Created indexes for interactions collection")

        # Indexes for metadata
        await self.db.metadata.create_index("data_type")
        await self.db.metadata.create_index("timestamp")
        logger.debug("Created indexes for metadata collection")

        # Indexes for feedback
        await self.db.feedback.create_index("session_id")
        await self.db.feedback.create_index("timestamp")
        logger.debug("Created indexes for feedback collection")

    async def log_interaction(self, session_id: str, query: str, complexity: Dict[str, Any],
                              budget_allocation: Dict[str, Any], performance_metrics: Dict[str, Any],
                              success: bool, error_message: Optional[str] = None):
        """Log interaction data to MongoDB"""
        doc = {
            "session_id": session_id,
            "query": query,
            "complexity": to_mongodb_serializable(complexity),
            "budget_allocation": to_mongodb_serializable(budget_allocation),
            "performance_metrics": to_mongodb_serializable(performance_metrics),
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.utcnow()
        }
        result = await self.db.interactions.insert_one(doc)
        logger.debug(f"Logged interaction {result.inserted_id}")
        return str(result.inserted_id)

    async def _store_to_mongodb(self, data_type: str, content: str, metadata: Dict) -> str:
        """Store document in MongoDB metadata collection"""
        doc = {
            "data_type": data_type,
            "content": to_mongodb_serializable(content),
            "metadata": to_mongodb_serializable(metadata),
            "timestamp": datetime.utcnow()
        }
        result = await self.db.metadata.insert_one(doc)
        logger.debug(f"Stored document in MongoDB with id {result.inserted_id}")
        return str(result.inserted_id)

    async def _store_metadata(self, data_type: str, content: str, metadata: Dict, destination: str) -> str:
        """Store metadata log in MongoDB"""
        meta_doc = {
            "data_type": data_type,
            "content_preview": content[:200],
            "destination": destination,
            "metadata": to_mongodb_serializable(metadata),
            "timestamp": datetime.utcnow(),
            "routing_decision": "automatic"
        }
        result = await self.db.metadata.insert_one(meta_doc)
        logger.debug(f"Stored metadata log in MongoDB with id {result.inserted_id}")
        return str(result.inserted_id)

    async def store_feedback(self, session_id: str, rating: int, comments: Optional[str] = None) -> str:
        """Store feedback in MongoDB"""
        doc = {
            "session_id": session_id,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.utcnow()
        }
        result = await self.db.feedback.insert_one(doc)
        logger.info(f"Stored feedback with id {result.inserted_id}")
        return str(result.inserted_id)

    async def close(self):
        """Clean up resources and terminate local mongod if started"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Closed MongoDB client")
        if self.mongod_process and self._started_local_mongo:
            self.mongod_process.terminate()
            await asyncio.sleep(1)
            logger.info("Terminated local mongod process")

    def __del__(self):
        """Ensure resources are cleaned up on object deletion"""
        asyncio.create_task(self.close())