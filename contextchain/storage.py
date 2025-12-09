"""
IntelligentStorage for ContextChain v2.0 — Now with Full Chat Memory
- SQLite-based persistent conversational memory
- Per-session + optional per-user long-term memory
- Exact short-term recall + semantic long-term retrieval ready
- Automatic token counting (optional, for future ACBA improvement)
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
    """Recursively convert objects to JSON-serializable format"""
    if isinstance(obj, Enum):
        return obj.value if hasattr(obj, 'value') else str(obj.name)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
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
        """Initialize SQLite connection and ensure all tables exist"""
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            await self.conn.execute("PRAGMA foreign_keys = ON;")
            await self._ensure_tables()
            logger.info(f"IntelligentStorage initialized → {self.db_path}")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {str(e)}")
            raise

    async def _ensure_tables(self):
        """Create all required tables with optimal indexes"""

        # Interactions (system analytics)
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

        # General metadata storage
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

        # Metadata routing logs
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

        # User feedback
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

        # ╔══════════════════════════════════════════════════════════╗
        # ║                   CHAT MEMORY TABLE                      ║
        # ╚══════════════════════════════════════════════════════════╝
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT,                                   -- Optional: enables cross-session memory
                role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')), 
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,                       -- For future ACBA precision
                metadata TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_messages(timestamp)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session_timestamp ON chat_messages(session_id, timestamp)")

        await self.conn.commit()
        logger.debug("All tables and indexes ensured (including chat_messages)")

    # ─────────────────────────── Chat Memory API ───────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,  # 'system', 'user', 'assistant'
        content: str,
        user_id: Optional[str] = None,
        tokens: Optional[int] = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store a single message and return its row ID"""
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO chat_messages 
            (session_id, user_id, role, content, tokens, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            user_id,
            role,
            content,
            tokens,
            json.dumps(to_json_serializable(metadata or {})),
            timestamp
        ))
        await self.conn.commit()
        return cur.lastrowid

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent messages in chronological order (oldest first)
        Perfect for injecting into the prompt
        """
        cur = await self.conn.cursor()
        await cur.execute("""
            SELECT role, content, tokens, metadata, timestamp
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit))
        rows = await cur.fetchall()
        # Reverse to get chronological order (oldest → newest)
        messages = [
            {
                "role": row[0],
                "content": row[1],
                "tokens": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "timestamp": row[4]
            }
            for row in reversed(rows)
        ]
        return messages

    async def get_conversation_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        since: Optional[str] = None,  # ISO timestamp
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Flexible history retrieval (for analytics, export, or debugging)"""
        query = "SELECT role, content, timestamp, tokens FROM chat_messages WHERE session_id = ?"
        params: List[Any] = [session_id]

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cur = await self.conn.cursor()
        await cur.execute(query, params)
        rows = await cur.fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2], "tokens": r[3]} for r in rows]

    async def clear_session_history(self, session_id: str) -> int:
        """Delete all messages for a session (useful for /reset commands)"""
        cur = await self.conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        await self.conn.commit()
        deleted = cur.rowcount
        logger.info(f"Cleared {deleted} messages for session {session_id}")
        return deleted

    # ─────────────────────────── Existing Methods (unchanged) ───────────────────────────

    async def log_interaction(self, session_id: str, query: str, complexity: Dict[str, Any],
                              budget_allocation: Dict[str, Any], performance_metrics: Dict[str, Any],
                              success: bool, error_message: Optional[str] = None):
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO interactions (session_id, query, complexity, budget_allocation, performance_metrics, success, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, query,
            json.dumps(to_json_serializable(complexity)),
            json.dumps(to_json_serializable(budget_allocation)),
            json.dumps(to_json_serializable(performance_metrics)),
            success, error_message, timestamp
        ))
        await self.conn.commit()
        return str(cur.lastrowid)

    async def _store_to_sqlite(self, data_type: str, content: str, metadata: Dict) -> str:
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO metadata (data_type, content, metadata, timestamp)
            VALUES (?, ?, ?, ?)
        """, (data_type, content, json.dumps(to_json_serializable(metadata)), timestamp))
        await self.conn.commit()
        return str(cur.lastrowid)

    async def _store_metadata(self, data_type: str, content: str, metadata: Dict, destination: str) -> str:
        timestamp = datetime.utcnow().isoformat()
        content_preview = content[:200]
        cur = await self.conn.execute("""
            INSERT INTO metadata_logs (data_type, content_preview, destination, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (data_type, content_preview, destination, json.dumps(to_json_serializable(metadata)), timestamp))
        await self.conn.commit()
        return str(cur.lastrowid)

    async def store_feedback(self, session_id: str, rating: int, comments: Optional[str] = None) -> str:
        timestamp = datetime.utcnow().isoformat()
        cur = await self.conn.execute("""
            INSERT INTO feedback (session_id, rating, comments, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, rating, comments, timestamp))
        await self.conn.commit()
        logger.info(f"Stored feedback for session {session_id} — rating: {rating}")
        return str(cur.lastrowid)

    async def close(self):
        if self.conn:
            await self.conn.close()
            logger.info("IntelligentStorage → SQLite connection closed")

    def __del__(self):
        if self.conn:
            asyncio.create_task(self.close())