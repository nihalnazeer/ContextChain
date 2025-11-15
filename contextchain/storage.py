"""
IntelligentStorage for ContextChain v2.0
- Manages SQLite connection lifecycle
- Creates and maintains necessary tables for interactions, metadata, and feedback
- Provides async methods for storing and querying data
- Designed for seamless integration with ContextChain core.py and API
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

import aiosqlite

logger = logging.getLogger(__name__)

def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable format.
    Handles Enum, datetime, and nested dataclasses/Pydantic models.
    """
    if isinstance(obj, Enum):
        return obj.value if hasattr(obj, 'value') else str(obj.name)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
        # Convert dataclasses, Pydantic models, or objects with __dict__
        return {k: to_json_serializable(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    else:
        return obj

class IntelligentStorage:
    def __init__(self, db_path: str = "contextchain.db"):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Initialize SQLite connection and ensure tables exist"""
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            await self.conn.execute("PRAGMA foreign_keys = ON;")
            await self._ensure_tables()
            logger.info(f"Connected to SQLite at {self.db_path}")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {str(e)}")
            raise

    async def _ensure_tables(self):
        """Ensure necessary tables exist with indexes"""
        # Interactions table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                complexity TEXT NOT NULL,
                budget_allocation TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")

        # Metadata table (for _store_to_sqlite)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_data_type ON metadata(data_type)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_timestamp ON metadata(timestamp)")

        # Metadata log table (for _store_metadata, separate for distinction)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                content_preview TEXT NOT NULL,
                destination TEXT NOT NULL,
                metadata TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                routing_decision TEXT DEFAULT 'automatic'
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_logs_data_type ON metadata_logs(data_type)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_logs_timestamp ON metadata_logs(timestamp)")

        # Feedback table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comments TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")

        await self.conn.commit()
        logger.debug("Ensured tables and indexes for SQLite")

    async def log_interaction(self, session_id: str, query: str, complexity: Dict[str, Any],
                              budget_allocation: Dict[str, Any], performance_metrics: Dict[str, Any],
                              success: bool, error_message: Optional[str] = None):
        """Log interaction data to SQLite"""
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO interactions (session_id, query, complexity, budget_allocation, performance_metrics, success, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            query,
            json.dumps(to_json_serializable(complexity)),
            json.dumps(to_json_serializable(budget_allocation)),
            json.dumps(to_json_serializable(performance_metrics)),
            success,
            error_message,
            timestamp
        ))
        await self.conn.commit()
        rowid = cur.lastrowid
        logger.debug(f"Logged interaction {rowid}")
        return str(rowid)

    async def _store_to_sqlite(self, data_type: str, content: str, metadata: Dict) -> str:
        """Store document in SQLite metadata collection"""
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO metadata (data_type, content, metadata, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            data_type,
            content,
            json.dumps(to_json_serializable(metadata)),
            timestamp
        ))
        await self.conn.commit()
        rowid = cur.lastrowid
        logger.debug(f"Stored document in SQLite with id {rowid}")
        return str(rowid)

    async def _store_metadata(self, data_type: str, content: str, metadata: Dict, destination: str) -> str:
        """Store metadata log in SQLite"""
        timestamp = datetime.utcnow().isoformat()
        content_preview = content[:200]
        cur = await self.conn.execute("""
            INSERT INTO metadata_logs (data_type, content_preview, destination, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data_type,
            content_preview,
            destination,
            json.dumps(to_json_serializable(metadata)),
            timestamp
        ))
        await self.conn.commit()
        rowid = cur.lastrowid
        logger.debug(f"Stored metadata log in SQLite with id {rowid}")
        return str(rowid)

    async def store_feedback(self, session_id: str, rating: int, comments: Optional[str] = None) -> str:
        """Store feedback in SQLite"""
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO feedback (session_id, rating, comments, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            session_id,
            rating,
            comments,
            timestamp
        ))
        await self.conn.commit()
        rowid = cur.lastrowid
        logger.info(f"Stored feedback with id {rowid}")
        return str(rowid)

    async def close(self):
        """Clean up resources"""
        if self.conn:
            await self.conn.close()
            logger.info("Closed SQLite connection")

    def __del__(self):
        """Ensure resources are cleaned up on object deletion"""
        if self.conn:
            asyncio.create_task(self.close())